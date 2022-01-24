import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

from utils import Rescale, cycle, grab_checkpoint
from diff_model import DiffusionModel
from diffusion import Diffusion


# ADD SMARTER BETA SCHEDULE SAMPLING
# ADD DISTRIBUTED TRAINING WITH TORCH.DIST
# ADD ANNEALED LR?

# class Dataset:
#     def __init__(self, resolution, batch_size, conditional, random_crop=False, random_flip=True):
#         self.batch_size = batch_size
#         self.conditional = conditional
#
#         # THIS IS FOR CUSTOM DATASETS, DO THIS LATER


class Trainer:
    def __init__(self, model, diffusion_args, dataloader, iterations,
                 batch_size, lr, weight_decay, ema_rate=0.9999,
                 grad_accumulation=1,
                 checkpoint=(None, None, None, None),
                 print_every=None, sample_every=None, save_every=None, device=None):
        # assert model.conditional == dataloader.conditional, 'model and dataloader should be either conditional or not'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = model
        self.model.to(self.device)

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
            self.start_step = checkpoint[3]
            self.ema_params = {name: ema_state_dict[name] for name, _ in model.named_parameters()}
        else:
            self.start_step = 0
            self.ema_params = {name: param.data for name, param in model.named_parameters()}
        for _, p in self.ema_params.items():
            p.requires_grad = False
            p.to(torch.device('cpu'))

        self.ema_rate = ema_rate
        self.iterations = iterations
        self.print_every = print_every
        self.sample_every = sample_every
        self.save_every = save_every

    def train(self):
        running_loss = torch.tensor([0], dtype=torch.float, device=self.device)
        for step in range(self.iterations):
            self.model.train()

            # Labels will be none if model is not conditional
            batch, labels = next(self.loader)

            # batch = torch.randn(self.batch_size, 3, self.model.resolution, self.model.resolution, device=self.device)
            # labels = torch.randint(low=0, high=self.model.num_classes, size=(self.batch_size,), device=self.device)
            batch = batch.permute(0, 1, 3, 2).to(self.device)  # added because EMNIST is in w,h instead of h,w
            labels = labels.to(self.device)

            # Change labels to null label with probability 1% during training if we are using classifier-free guidance
            if labels is not None:
                if self.train_diffusion.guidance == 'classifier_free' and np.random.randint(100) < 2:
                    labels = torch.full_like(labels, fill_value=0).to(self.device)
                kwargs = {'y': labels}
            else:
                kwargs = None

            # Grab random timestep to calculate loss with
            t = torch.randint(low=0, high=self.train_diffusion.original_num_steps, size=(self.batch_size,),
                              device=self.device)

            loss = self.train_diffusion.loss(x_0=batch, t=t, kwargs=kwargs)
            loss = loss.mean() / self.grad_accumulation
            if (step + 1) % self.grad_accumulation == 0:
                loss.backward()
                self.opt.step()
                self.opt.zero_grad(set_to_none=False)
            running_loss += loss

            # update EMA -- ema <= rate * ema + (1 - rate) * current
            for name, param in model.named_parameters():
                self.ema_params[name].detach().mul_(self.ema_rate).add_(param.data, alpha=(1 - self.ema_rate))

            if self.print_every is not None and step % self.print_every == 0:
                print('Step #{}  ------------------------------------------\n\tLoss={}'.
                      format(step, running_loss.item() / self.print_every))
                running_loss = torch.tensor([0], dtype=torch.float, device=self.device)

            if self.sample_every is not None and step % self.sample_every == 0:
                self.sample(4)

            if self.save_every is not None and step % self.save_every == 0:
                self.save(self.start_step + step)

        self.save(self.start_step + self.iterations)
        pass

    def sample(self, num_samples):
        self.model.eval()
        labels = torch.randint(low=0, high=self.model.num_classes, size=(num_samples,),
                               device=self.device, requires_grad=False)
        out = self.sampling_diffusion.denoise(kwargs={'y': labels}, batch_size=num_samples, ema_params=self.ema_params)
        out = ((out + 1) * 127.5).clamp(0, 255)
        out = out.to(torch.uint8).permute(0, 2, 3, 1).cpu().detach().numpy()
        for b in range(num_samples):
            plt.close('all')
            if self.model.in_channels == 1:
                plt.imshow(out[b].astype(np.uint8), cmap='Greys')
            else:
                plt.imshow(out[b].astype(np.uint8))
            plt.title('Label: {}'.format(labels[b].cpu().item()))
            plt.pause(0.0001)
            plt.waitforbuttonpress()
        plt.close('all')
        pass

    def save(self, step):
        torch.save(self.model.state_dict(), 'checkpoints/{}_model_params.pt'.format(step))
        torch.save(self.opt.state_dict(), 'checkpoints/{}_opt_params.pt'.format(step))
        torch.save(self.ema_params, 'checkpoints/{}_ema_params.pt'.format(step))
        print('Saved checkpoint!')
        pass


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # HYPERPARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    # torch.manual_seed(0)
    CONDITIONAL = True
    DIFFUSION_ARGS = {'rescaled_num_steps': 1000, 'original_num_steps': 1000, 'use_ddim': False, 'ddim_eta': 0.0,
                      'beta_schedule': 'cosine', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                      'guidance_method': 'classifier_free', 'guidance_strength': 0.5, 'loss_type': 'hybrid'}
    MODEL_ARGS = {'resolution': 28, 'attention_resolutions': (7, 14), 'channel_mult': (1, 2, 4),
                  'num_heads': 4, 'in_channels': 1, 'out_channels': 2, 'model_channels': 64,
                  'num_res_blocks': 2, 'split_qkv_first': True, 'dropout': 0.05,
                  'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 26 if CONDITIONAL else None}

    BATCH_SIZE = 468
    LR = 0.00016
    WEIGHT_DECAY = 0.001

    ITERATIONS = 1500
    SAMPLE_EVERY = None
    PRINT_EVERY = 10
    SAVE_EVERY = 100

    GRAD_ACCUMULATION = 1

    # CHECKPOINT = grab_checkpoint(20000)
    CHECKPOINT = (None, None, None, None)
    # ------------------------------------------------------------------------------------------------------------------

    if DIFFUSION_ARGS['guidance_method'] == 'classifier_free':
        MODEL_ARGS['num_classes'] += 1

    model = DiffusionModel(**MODEL_ARGS, use_grad_checkpoints=True)
    print('Model has {} parameters'.format(sum(p.numel() for p in model.parameters())))

    transform = transforms.Compose([transforms.ToTensor(), Rescale()])
    train_data = EMNIST(root='datasets/', train=True, download=False, transform=transform, split='letters')
    loader = cycle(DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True, num_workers=4))
    trainer = Trainer(model=model, diffusion_args=DIFFUSION_ARGS, dataloader=loader, ema_rate=0.9999,
                      batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
                      iterations=ITERATIONS, sample_every=SAMPLE_EVERY, print_every=PRINT_EVERY, save_every=SAVE_EVERY,
                      grad_accumulation=GRAD_ACCUMULATION, checkpoint=CHECKPOINT)
    trainer.train()
