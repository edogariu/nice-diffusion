from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms

from nicediffusion.model import DiffusionModel
from nicediffusion.trainer import Trainer
from nicediffusion.utils import Rescale, cycle
from nicediffusion.default_args import EMNIST_DIFFUSION_ARGS, EMNIST_MODEL_ARGS
# ------------------------------------------------------------------------------------------------------------------
# HYPERPARAMETERS
# ------------------------------------------------------------------------------------------------------------------
# torch.manual_seed(0)
# CONDITIONAL = True
# DIFFUSION_ARGS = {'rescaled_num_steps': 1000, 'original_num_steps': 1000, 'use_ddim': False, 'ddim_eta': 0.0,
#                   'beta_schedule': 'cosine', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
#                   'guidance_method': 'classifier_free', 'guidance_strength': 0.5, 'loss_type': 'hybrid'}
# MODEL_ARGS = {'resolution': 28, 'attention_resolutions': (7, 14), 'channel_mult': (1, 2, 4),
#               'num_heads': 4, 'in_channels': 1, 'out_channels': 2, 'model_channels': 64,
#               'num_res_blocks': 2, 'split_qkv_first': True, 'dropout': 0.05,
#               'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 26 if CONDITIONAL else None}
DIFFUSION_ARGS = EMNIST_DIFFUSION_ARGS
MODEL_ARGS = EMNIST_MODEL_ARGS

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