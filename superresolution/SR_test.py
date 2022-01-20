import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from basicsr.archs.rrdbnet_arch import RRDBNet  # https://github.com/xinntao/BasicSR

from swin import SwinIR
from diff_model import SuperResolutionModel, convert_state_dict
from diffusion import Diffusion

"""
Script to compare various SuperResolution models. The plan is to use one to upscale the diffusion outputs in a 
cascading-type setup.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

upscale = 4
start_size = 128

IM_PATH = 'images/archive/images/images/Claude_Monet/Claude_Monet_18.jpg'
# IM_PATH = 'cuteimages/forest.jpg'

# METHODS_TO_TEST = ['ESRGAN', 'SWIN', 'Diffusion', 'classic']
METHODS_TO_TEST = ['ESRGAN', 'SWIN', 'classic']

image = cv2.imread(IM_PATH, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
ground_truth = cv2.cvtColor(cv2.resize(image, dsize=(start_size * upscale, start_size * upscale)) * 255,
                            cv2.COLOR_BGR2RGB).astype('uint8')
image = cv2.resize(image, dsize=(start_size, start_size))

# ---------------------------------------------------------------------------------------------------------------------
# UPSAMPLE WITH Real-ESRGAN
# (all credits to https://github.com/xinntao/Real-ESRGAN, https://arxiv.org/pdf/1809.00219.pdf)
# ---------------------------------------------------------------------------------------------------------------------
if 'ESRGAN' in METHODS_TO_TEST:
    if upscale == 4:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model.load_state_dict(torch.load('models/RealESRGAN_x4plus.pth')['params_ema'], strict=True)
    else:
        raise NotImplementedError(str(upscale) + 'x ESRGAN')
    model.float().to(device)
    model.eval()

    print('Doing SR with ESRGAN model with {} parameters!'.format(sum(p.numel() for p in model.parameters())))

    img_lq = image
    img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    output = model(img_lq)
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))  # CHW-RGB to HCW-RGB
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    output_esrgan = output
else:
    output_esrgan = None

# ---------------------------------------------------------------------------------------------------------------------
# UPSAMPLE WITH SWIN
# (all credits to https://github.com/JingyunLiang/SwinIR)
# ---------------------------------------------------------------------------------------------------------------------
if 'SWIN' in METHODS_TO_TEST:
    if upscale == 4:
        model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                       num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                       mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        model.load_state_dict(torch.load(
            'models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR-with-dict-keys-params-and-params_ema.pth'.format(
                upscale))['params_ema'], strict=True)
    elif upscale == 8:
        model = SwinIR(upscale=8, in_chans=3, img_size=48, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        model.load_state_dict(torch.load(
            'models/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth'.format(upscale))['params'], strict=True)
    else:
        raise NotImplementedError(str(upscale) + 'x SWIN')
    model.float().to(device)
    model.eval()
    print('Doing SR with SWIN model with {} parameters!'.format(sum(p.numel() for p in model.parameters())))

    img_lq = image
    img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // 8 + 1) * 8 - h_old
        w_pad = (w_old // 8 + 1) * 8 - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        output = output[..., :h_old * upscale, :w_old * upscale]

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))  # CHW-RGB to HCW-RGB
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    output_swin = output
else:
    output_swin = None

# ---------------------------------------------------------------------------------------------------------------------
# UPSAMPLE THE GOOD OL WAY
# ---------------------------------------------------------------------------------------------------------------------
if 'classic' in METHODS_TO_TEST:
    im = np.transpose(image, (2, 0, 1))[[2, 1, 0], :, :].astype(np.float32)  # HWC-BGR -> CHW-RGB
    im = torch.from_numpy(im).float().unsqueeze(0).to(device=device)

    mode = 'bicubic'
    interpolated = F.interpolate(im, scale_factor=upscale, mode=mode, align_corners=False)
    output = interpolated.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))  # CHW-RGB -> HWC-RGB
    output = (output * 255.0).round().astype(np.uint8)
    output_classic = output
else:
    output_classic = None
    mode = None

# ---------------------------------------------------------------------------------------------------------------------
# UPSAMPLE WITH SuperResolutionModel
# (all credits to https://github.com/openai/guided-diffusion and my reimplementation)
# ---------------------------------------------------------------------------------------------------------------------
if 'Diffusion' in METHODS_TO_TEST:
    DIFF_ARGS = {'beta_schedule': 'linear', 'sampling_var_type': 'learned_range', 'classifier': None,
                 'guidance_method': None, 'guidance_strength': None, 'device': device,
                 'rescaled_num_steps': 25, 'original_num_steps': 1000, 'use_ddim': True, 'ddim_eta': 0.0}
    MODEL_ARGS = {'upscale_resolution': 512, 'attention_resolutions': (16, 32), 'channel_mult': (1, 1, 2, 2, 4, 4),
                  'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 192,
                  'num_res_blocks': 2, 'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000}
    model = SuperResolutionModel(**MODEL_ARGS)
    model.load_state_dict(convert_state_dict(torch.load(
        'models/128_512_upsampler.pt', map_location="cpu")), strict=True)
    model.eval()
    diff = Diffusion(model, **DIFF_ARGS)
    print('Doing SR with Diffusion model with {} parameters!'.format(sum(p.numel() for p in model.parameters())))

    im = image
    im = np.transpose(im[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
    im = torch.from_numpy(im).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    sample = diff.denoise(x=torch.randn(size=(1, 3, 512, 512)).to(device),
                          kwargs={'y': torch.full(size=(1,), fill_value=975, device=device), 'low_res': im})
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.contiguous().detach().squeeze().cpu().numpy().astype('uint8')
    sample = np.transpose(sample, (1, 2, 0))
    output_diff = sample
else:
    output_diff = None

# ---------------------------------------------------------------------------------------------------------------------
# PLOT EM ALL
# ---------------------------------------------------------------------------------------------------------------------
if 'ESRGAN' in METHODS_TO_TEST:
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(ground_truth)
    plt.title('Ground Truth')
    plt.pause(0.001)

    fig.add_subplot(1, 2, 2)
    plt.imshow(output_esrgan)
    plt.title('ESRGAN')
    plt.pause(0.001)
    plt.waitforbuttonpress(0)

if 'Diffusion' in METHODS_TO_TEST:
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(ground_truth)
    plt.title('Ground Truth')
    plt.pause(0.001)

    fig.add_subplot(1, 2, 2)
    plt.imshow(output_diff)
    plt.title('Diffusion')
    plt.pause(0.001)
    plt.waitforbuttonpress(0)

if 'SWIN' in METHODS_TO_TEST:
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(ground_truth)
    plt.title('Ground Truth')
    plt.pause(0.001)

    fig.add_subplot(1, 2, 2)
    plt.imshow(output_swin)
    plt.title('SWIN')
    plt.pause(0.001)
    plt.waitforbuttonpress(0)

if 'classic' in METHODS_TO_TEST:
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(ground_truth)
    plt.title('Ground Truth')
    plt.pause(0.001)

    fig.add_subplot(1, 2, 2)
    plt.imshow(output_classic)
    plt.title(mode)
    plt.pause(0.001)
    plt.waitforbuttonpress(0)
