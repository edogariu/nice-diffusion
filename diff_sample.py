import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from basicsr.archs.rrdbnet_arch import RRDBNet

from diff_model import DiffusionModel, convert_state_dict
from diffusion import Diffusion

# ---------------------------------------------------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
# torch.manual_seed(0)

STATE_DICT_FILENAME = 'checkpoints/3000_model_params.pt'
# STATE_DICT_FILENAME = 'models/64x64_diffusion.pt'
# STATE_DICT_FILENAME = 'models/128x128_diffusion.pt'
# STATE_DICT_FILENAME = 'models/256x256_diffusion_uncond.pt'

DIFFUSION_ARGS = {'rescaled_num_steps': 250, 'original_num_steps': 1000, 'use_ddim': False, 'ddim_eta': 0.5}

BATCH_SIZE = 16
NUM_SAMPLES = 1
DESIRED_LABELS = [] * NUM_SAMPLES  # set to list of labels (one for each sample) or [] for random label each sample

# Image filepath and number of diffusion steps to apply to it to get a starting point for denoising.
# STEPS_TO_DO is out of original_num_steps, regardless of whether we use rescaled or DDIM sampling
START_IMG, STEPS_TO_DO = None, 400

SHOW_PROGRESS = True
UPSAMPLE = False  # Whether to 4x upsample generated image with Real-ESRGAN (https://github.com/xinntao/Real-ESRGAN)

# NOTE: classifier not implemented yet since I have not yet trained a noisy classifier
GUIDANCE = None  # can be None, 'classifier', or 'classifier_free'
GUIDANCE_STRENGTH = 1.0 if GUIDANCE is not None else None

# ---------------------------------------------------------------------------------------------------------------------

# CREATE MODEL(S)
if GUIDANCE == 'classifier':
    raise NotImplementedError('sorry, i haven\'t yet trained a noisy classifier! :)')

if STATE_DICT_FILENAME.__contains__('_params.pt'):
    CONDITIONAL = True
    DIFF_ARGS = {'beta_schedule': 'cosine', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                 'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
                 'loss_type': 'hybrid', 'device': device}
    MODEL_ARGS = {'resolution': 28, 'attention_resolutions': (7, 14), 'channel_mult': (1, 2, 4),
                  'num_heads': 4, 'in_channels': 1, 'out_channels': 2, 'model_channels': 64,
                  'num_res_blocks': 2, 'split_qkv_first': True, 'dropout': 0.05,
                  'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 10 if CONDITIONAL else None}
    state_dict = torch.load(STATE_DICT_FILENAME, map_location="cpu")
elif STATE_DICT_FILENAME == 'models/64x64_diffusion.pt':
    CONDITIONAL = True
    DIFF_ARGS = {'beta_schedule': 'cosine', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                 'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
                 'loss_type': 'hybrid', 'device': device}
    MODEL_ARGS = {'resolution': 64, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 2, 3, 4),
                  'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 192,
                  'num_res_blocks': 3, 'split_qkv_first': True,
                  'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000 if CONDITIONAL else None}
    state_dict = convert_state_dict(torch.load(STATE_DICT_FILENAME, map_location="cpu"))
elif STATE_DICT_FILENAME == 'models/128x128_diffusion.pt':
    CONDITIONAL = True
    DIFF_ARGS = {'beta_schedule': 'linear', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                 'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
                 'loss_type': 'hybrid', 'device': device}
    MODEL_ARGS = {'resolution': 128, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 1, 2, 3, 4),
                  'num_heads': 4, 'in_channels': 3, 'out_channels': 6, 'model_channels': 256,
                  'num_res_blocks': 2, 'split_qkv_first': False,
                  'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000 if CONDITIONAL else None}
    state_dict = convert_state_dict(torch.load(STATE_DICT_FILENAME, map_location="cpu"))
elif STATE_DICT_FILENAME == 'models/256x256_diffusion_uncond.pt':
    CONDITIONAL = False
    DIFF_ARGS = {'beta_schedule': 'linear', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                 'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
                 'loss_type': 'hybrid', 'device': device}
    MODEL_ARGS = {'resolution': 256, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 1, 2, 2, 4, 4),
                  'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 256,
                  'num_res_blocks': 2, 'split_qkv_first': False,
                  'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000 if CONDITIONAL else None}
    state_dict = convert_state_dict(torch.load(STATE_DICT_FILENAME, map_location="cpu"))
else:
    raise NotImplementedError(STATE_DICT_FILENAME)

model = DiffusionModel(**MODEL_ARGS)
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()

print('Model made from {} with {} parameters! :)\n'.
      format(STATE_DICT_FILENAME, sum(p.numel() for p in model.parameters())))

print('Starting Diffusion! There are {} samples of {} images each\n'.format(NUM_SAMPLES, BATCH_SIZE))
samples = []
DIFFUSION_ARGS.update(DIFF_ARGS)
diffusion = Diffusion(model=model, **DIFFUSION_ARGS)

if START_IMG is not None and STEPS_TO_DO is not None:
    START_IMG = cv2.imread(START_IMG)
    # may want to replace this resize with a center crop?
    START_IMG = cv2.resize(START_IMG, dsize=(MODEL_ARGS['resolution'], MODEL_ARGS['resolution'])) / 127.5 - 1
    START_IMG = torch.from_numpy(START_IMG).permute(2, 0, 1)[[2, 1, 0], :, :]
    start_imgs = torch.zeros(size=(BATCH_SIZE, MODEL_ARGS['in_channels'],
                                   MODEL_ARGS['resolution'], MODEL_ARGS['resolution']))
    for i_batch in range(BATCH_SIZE):
        temp = START_IMG.clone()
        start_imgs[i_batch] = temp
    START_IMG = start_imgs.to(device)


if CONDITIONAL and len(DESIRED_LABELS) != 0:
    assert len(DESIRED_LABELS) == NUM_SAMPLES, 'please provide NUM_SAMPLES={} labels'.format(NUM_SAMPLES)
for i_sample in range(NUM_SAMPLES):
    # CREATE RANDOM DATA
    if START_IMG is None or STEPS_TO_DO is None:
        data = torch.randn([BATCH_SIZE, MODEL_ARGS['in_channels'],
                            MODEL_ARGS['resolution'], MODEL_ARGS['resolution']]).to(device)
        steps = DIFFUSION_ARGS['rescaled_num_steps']
    else:
        steps = STEPS_TO_DO * DIFFUSION_ARGS['rescaled_num_steps'] // DIFFUSION_ARGS['original_num_steps']
        data = diffusion.diffuse(x_0=START_IMG, steps_to_do=steps)
    if CONDITIONAL:
        if len(DESIRED_LABELS) == 0:
            labels = torch.randint(low=0, high=MODEL_ARGS['num_classes'], size=(BATCH_SIZE,), device=device)
        else:
            labels = torch.full(size=(BATCH_SIZE,), fill_value=DESIRED_LABELS[i_sample], device=device)
    else:
        labels = None

    # RUN DIFFUSION
    print('Denoising sample {}! :)'.format(i_sample + 1))
    out = diffusion.denoise(x=data, kwargs={'y': labels}, batch_size=BATCH_SIZE, progress=SHOW_PROGRESS,
                            steps_to_do=steps)

    # Convert from [-1.0, 1.0] to [0, 255]
    out = ((out + 1) * 127.5).clamp(0, 255)
    data = ((data + 1) * 127.5).clamp(0, 255)
    if START_IMG is not None:
        START_IMG = ((START_IMG + 1) * 127.5).clamp(0, 255)
        samples.append((START_IMG.cpu(), out.cpu()))
    else:
        samples.append((data.cpu(), out.cpu(), labels.cpu()))
    print()


# Show image (img must be RGB and from [0.0, 1.0] or [0, 255])
def imshow(img, title=None):
    plt.imshow(img.astype(np.uint8))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


print('Displaying {} generated images!'.format(NUM_SAMPLES * BATCH_SIZE))
if UPSAMPLE:
    print('Upsampling to {}x{} resolution!'.format(MODEL_ARGS['resolution'] * 4, MODEL_ARGS['resolution'] * 4))
    model.to(torch.device('cpu'))  # deallocate diffusion model memory
    del model
    # avoid cuda alloc error on my 6GB GPU
    if (MODEL_ARGS['resolution'] > 64 and BATCH_SIZE > 1) or not torch.cuda.is_available():
        upsampling_device = torch.device('cpu')
    else:
        upsampling_device = torch.device('cuda')
    esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    esrgan.load_state_dict(torch.load('models/RealESRGAN_x4plus.pth', map_location=upsampling_device)['params_ema'],
                           strict=True)
    esrgan.to(upsampling_device).eval()

    upscaled_samples = []
    for sample in samples:
        data, out, labels = sample
        data = F.interpolate(data, scale_factor=4, mode='bilinear', align_corners=False)
        out = (out / 255.0).to(upsampling_device)
        out = esrgan(out).cpu() * 255.0
        out = out.clamp(0, 255)
        upscaled_samples.append((data, out, labels))
    samples = upscaled_samples

for sample in samples:
    data, out, labels = sample
    # Convert from NCHW-RGB to HWC-RGB
    data = data.permute(0, 2, 3, 1).detach().numpy()
    out = out.to(torch.uint8).permute(0, 2, 3, 1).detach().numpy()
    for b in range(BATCH_SIZE):
        plt.close('all')
        fig = plt.figure(figsize=(7, 3))
        fig.add_subplot(1, 2, 1)
        imshow(data[b], title='Input Noise')
        fig.add_subplot(1, 2, 2)
        imshow(out[b], title='Output Image, Label={}'.format(labels[b].detach().numpy()))
        plt.waitforbuttonpress()
