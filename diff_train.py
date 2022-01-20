import time

import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

from diff_model import DiffusionModel
from diffusion import Diffusion


# ADD SMARTER BETA SCHEDULE SAMPLING
# ADD DISTRIBUTED TRAINING WITH TORCH.DIST
# ADD ANNEALED LR?

class DataLoader:
    def __init__(self, resolution, batch_size, conditional, random_crop=False, random_flip=True):
        self.batch_size = batch_size
        self.conditional = conditional

        # DO THIS LATER


class Trainer:
    def __init__(self, model, diffusion_args, dataloader,
                 iterations, batch_size, lr, weight_decay, ema_rate=0.9999, use_fp16=False, grad_accumulation=1,
                 checkpoint=(None, None, None, None),
                 print_every=None, test_every=None, save_every=None, device=None):
        assert model.conditional == dataloader.conditional, 'model and dataloader should be either conditional or not'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = model
        self.model.to(self.device)

        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        if self.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        self.grad_accumulation = grad_accumulation

        self.train_diffusion = Diffusion(**diffusion_args, model=model)
        diffusion_args.update({'rescaled_num_steps': 250, 'use_ddim': False})
        self.sampling_diffusion = Diffusion(**diffusion_args, model=model)

        self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

        self.loader = dataloader
        self.batch_size = batch_size

        # checkpoint is tuple of model_params path, ema_params path, optimizer_params path, and training step we are
        # resuming with
        if any(c is not None for c in checkpoint):
            assert not checkpoint.__contains__(None), \
                'please provide model, ema, and optimizer checkpoint paths and number of steps to resume with'
            self.model.load_state_dict(torch.load(checkpoint[0], map_location=device), strict=True)
            ema_state_dict = torch.load(checkpoint[1], map_location=device)
            self.opt.load_state_dict(torch.load(checkpoint[2], map_location=device))
            self.curr_step = checkpoint[3]
            self.ema_params = {name: ema_state_dict[name] for name, _ in model.named_parameters()}
        else:
            self.curr_step = 0
            self.ema_params = {name: param.data for name, param in model.named_parameters()}
        for _, p in self.ema_params.items():
            p.requires_grad = False
            p.to(torch.device('cpu'))

        self.ema_rate = ema_rate
        self.iterations = iterations
        self.print_every = print_every
        self.test_every = test_every
        self.save_every = save_every

    def train(self):
        start = time.time()
        running_loss = torch.tensor([0], dtype=torch.float, device=self.device)
        for step in range(self.iterations):
            self.model.train()

            # Labels will be none if model is not conditional
            # batch, labels = next(self.loader)
            batch = torch.randn(self.batch_size, 3, self.model.resolution, self.model.resolution, device=self.device)
            labels = torch.randint(low=0, high=self.model.num_classes, size=(self.batch_size,), device=self.device)
            batch = batch.to(self.device)
            labels = labels.to(self.device)

            # Change labels to 0 with probability 2% during training if we are using classifier-free guidance
            if labels is not None:
                if self.train_diffusion.guidance == 'classifier_free' and np.random.randint(100) < 2:
                    labels = torch.zeros_like(labels).to(self.device)
                kwargs = {'y': labels}
            else:
                kwargs = None

            # Grab random timestep to calculate loss with
            t = torch.randint(low=0, high=self.train_diffusion.original_num_steps, size=(self.batch_size,),
                              device=self.device)

            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    loss = self.train_diffusion.loss(x_0=batch, t=t, kwargs=kwargs)
                    loss = loss.mean()
                self.scaler.scale(loss / self.grad_accumulation).backward()
                if (step + 1) % self.grad_accumulation == 0:
                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.update()
            else:
                loss = self.train_diffusion.loss(x_0=batch, t=t, kwargs=kwargs)
                loss = loss.mean() / self.grad_accumulation
                if (step + 1) % self.grad_accumulation == 0:
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
            running_loss += loss

            # update EMA -- ema <= rate * ema + (1 - rate) * current
            for name, param in model.named_parameters():
                self.ema_params[name].detach().mul_(self.ema_rate).add_(param.data, alpha=(1 - self.ema_rate))

            if self.print_every is not None and step % self.print_every == 0:
                print('Step #{}  ------------------------------------------\n\tLoss={}'.
                      format(step, running_loss.item() / self.print_every))
                running_loss = torch.tensor([0], dtype=torch.float, device=self.device)

            if self.test_every is not None and step % self.test_every == 0:
                self.sample()

            if self.save_every is not None and step % self.save_every == 0:
                self.save()
        print(time.time() - start)
        pass

    def test(self):
        # DO THIS
        pass

    def sample(self):
        self.model.eval()
        out = self.sampling_diffusion.denoise(kwargs={'y': torch.randint(low=1, high=self.model.num_classes,
                                                                         size=(self.batch_size,), device=self.device)},
                                              batch_size=self.batch_size, ema_params=self.ema_params)
        out = ((out + 1) * 127.5).clamp(0, 255)
        out = out.to(torch.uint8).permute(0, 2, 3, 1).cpu().detach().numpy()
        for b in range(self.batch_size):
            plt.close('all')
            plt.imshow(out[b].astype(np.uint8))
            plt.pause(0.0001)
            plt.waitforbuttonpress()
        pass

    def save(self):
        # DO THIS
        pass


if __name__ == '__main__':
    # torch.manual_seed(0)
    CONDITIONAL = True
    DIFFUSION_ARGS = {'rescaled_num_steps': 4000, 'original_num_steps': 4000, 'use_ddim': False, 'ddim_eta': 0.0,
                      'beta_schedule': 'linear', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                      'guidance_method': None, 'guidance_strength': None, 'loss_type': 'hybrid'}
    MODEL_ARGS = {'resolution': 64, 'attention_resolutions': (8, 16), 'channel_mult': (1, 2, 3, 4),
                  'num_heads': 4, 'in_channels': 3, 'out_channels': 6, 'model_channels': 128,
                  'num_res_blocks': 3, 'split_qkv_first': True, 'dropout': 0.05,
                  'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 19 if CONDITIONAL else None}

    if DIFFUSION_ARGS['guidance_method'] == 'classifier_free':
        MODEL_ARGS['num_classes'] += 1

    model = DiffusionModel(**MODEL_ARGS)
    print('Model has {} parameters'.format(sum(p.numel() for p in model.parameters())))
    dataloader = DataLoader(128, 8, True)
    trainer = Trainer(model=model, diffusion_args=DIFFUSION_ARGS, dataloader=dataloader, ema_rate=0.9999,
                      iterations=50, batch_size=10, lr=0.0001, weight_decay=0.001, test_every=None, print_every=10,
                      use_fp16=False, grad_accumulation=1)
    trainer.train()
