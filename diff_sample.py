import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from basicsr.archs.rrdbnet_arch import RRDBNet

from utils import make_argparser, get_dicts_from_args
from diff_model import DiffusionModel
from diffusion import Diffusion


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = make_argparser('diff_sample')
    args = parser.parse_args()
    other_args, model_args, diff_args = get_dicts_from_args(args)

    # Gather hyperparameters
    if other_args['seed'] is not None:
        torch.manual_seed(other_args['seed'])
    WORDY = other_args['wordy']
    NUM_SAMPLES = other_args['num_samples']
    BATCH_SIZE = other_args['batch_size']
    UPSAMPLE = other_args['upsample']
    START_IMG, STEPS_TO_DO = other_args['start_img'], other_args['steps_to_do']
    LABELS = other_args['labels']
    CONDITIONAL = (model_args['num_classes'] is not None)

    model = DiffusionModel(**model_args)
    model.load_state_dict(torch.load(other_args['model_path'], map_location='cpu'), strict=True)
    model.to(device).eval()

    if WORDY:
        print('Model made from {} with {} parameters! :)\n'.
              format(other_args['model_path'], sum(p.numel() for p in model.parameters())))

        print('Starting Diffusion! There are {} samples of {} images each\n'.format(NUM_SAMPLES, BATCH_SIZE))
    samples = []
    diffusion = Diffusion(model=model, **diff_args)

    if START_IMG is not None and STEPS_TO_DO is not None:
        START_IMG = cv2.imread(START_IMG)
        # may want to replace this resize with a center crop?
        START_IMG = cv2.resize(START_IMG, dsize=(model_args['resolution'], model_args['resolution'])) / 127.5 - 1
        START_IMG = torch.from_numpy(START_IMG).permute(2, 0, 1)[[2, 1, 0], :, :]
        start_imgs = torch.zeros(size=(BATCH_SIZE, model_args['in_channels'],
                                       model_args['resolution'], model_args['resolution']))
        for i_batch in range(BATCH_SIZE):
            temp = START_IMG.clone()
            start_imgs[i_batch] = temp
        START_IMG = start_imgs.to(device)

    if CONDITIONAL and len(LABELS) != 0:
        assert len(LABELS) == NUM_SAMPLES, 'please provide NUM_SAMPLES={} labels'.format(NUM_SAMPLES)
    for i_sample in range(NUM_SAMPLES):
        # CREATE RANDOM DATA
        if START_IMG is None or STEPS_TO_DO is None:
            data = torch.randn([BATCH_SIZE, model_args['in_channels'],
                                model_args['resolution'], model_args['resolution']]).to(device)
            steps = diff_args['rescaled_num_steps']
        else:
            steps = STEPS_TO_DO * diff_args['rescaled_num_steps'] // diff_args['original_num_steps']
            data = diffusion.diffuse(x_0=START_IMG, steps_to_do=steps)
        if CONDITIONAL:
            if len(LABELS) == 0:
                labels = torch.randint(low=0, high=model_args['num_classes'], size=(BATCH_SIZE,), device=device)
            else:
                labels = torch.full(size=(BATCH_SIZE,), fill_value=LABELS[i_sample], device=device)
        else:
            labels = None

        # RUN DIFFUSION
        if WORDY:
            print('Denoising sample {}! :)'.format(i_sample + 1))
        out = diffusion.denoise(x=data, kwargs={'y': labels}, batch_size=BATCH_SIZE, progress=WORDY,
                                steps_to_do=steps)

        # Convert from [-1.0, 1.0] to [0, 255]
        out = ((out + 1) * 127.5).clamp(0, 255)
        data = ((data + 1) * 127.5).clamp(0, 255)
        if START_IMG is not None:
            START_IMG = ((START_IMG + 1) * 127.5).clamp(0, 255)
            samples.append((START_IMG.cpu(), out.cpu()))
        else:
            samples.append((data.cpu(), out.cpu(), labels.cpu()))
        if WORDY:
            print()

    # Show image (img must be RGB and from [0.0, 1.0] or [0, 255])
    def imshow(img, title=None):
        plt.imshow(img.astype(np.uint8))
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    if WORDY:
        print('Displaying {} generated images!'.format(NUM_SAMPLES * BATCH_SIZE))
    if UPSAMPLE:
        if WORDY:
            print('Upsampling to {}x{} resolution!'.format(model_args['resolution'] * 4, model_args['resolution'] * 4))
        model.to(torch.device('cpu'))  # deallocate diffusion model memory
        del model
        # avoid cuda alloc error on my 6GB GPU
        if (model_args['resolution'] > 64 and BATCH_SIZE > 1) or not torch.cuda.is_available():
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


if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------------------------------------------
# CODE STASH SO I REMEMBER THE HYPERPARAMETERS FOR SOME OF THE PRETRAINED MODELS
# ------------------------------------------------------------------------------------------------------------------
# # CREATE MODEL(S)
# if GUIDANCE == 'classifier':
#     raise NotImplementedError('sorry, i haven\'t yet trained a noisy classifier! :)')
#
# if STATE_DICT_FILENAME.__contains__('_params.pt'):
#     CONDITIONAL = True
#     DIFF_ARGS = {'beta_schedule': 'cosine', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
#                  'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
#                  'loss_type': 'hybrid', 'device': device}
#     MODEL_ARGS = {'resolution': 28, 'attention_resolutions': (7, 14), 'channel_mult': (1, 2, 4),
#                   'num_heads': 4, 'in_channels': 1, 'out_channels': 2, 'model_channels': 64,
#                   'num_res_blocks': 2, 'split_qkv_first': True, 'dropout': 0.05,
#                   'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 27 if CONDITIONAL else None}
#     state_dict = torch.load(STATE_DICT_FILENAME, map_location="cpu")
# elif STATE_DICT_FILENAME == 'models/64x64_diffusion.pt':
#     CONDITIONAL = True
#     DIFF_ARGS = {'beta_schedule': 'cosine', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
#                  'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
#                  'loss_type': 'hybrid', 'device': device}
#     MODEL_ARGS = {'resolution': 64, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 2, 3, 4),
#                   'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 192,
#                   'num_res_blocks': 3, 'split_qkv_first': True,
#                   'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000 if CONDITIONAL else None}
#     state_dict = convert_state_dict(torch.load(STATE_DICT_FILENAME, map_location="cpu"))
# elif STATE_DICT_FILENAME == 'models/128x128_diffusion.pt':
#     CONDITIONAL = True
#     DIFF_ARGS = {'beta_schedule': 'linear', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
#                  'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
#                  'loss_type': 'hybrid', 'device': device}
#     MODEL_ARGS = {'resolution': 128, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 1, 2, 3, 4),
#                   'num_heads': 4, 'in_channels': 3, 'out_channels': 6, 'model_channels': 256,
#                   'num_res_blocks': 2, 'split_qkv_first': False,
#                   'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000 if CONDITIONAL else None}
#     state_dict = convert_state_dict(torch.load(STATE_DICT_FILENAME, map_location="cpu"))
# elif STATE_DICT_FILENAME == 'models/256x256_diffusion_uncond.pt':
#     CONDITIONAL = False
#     DIFF_ARGS = {'beta_schedule': 'linear', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
#                  'guidance_method': GUIDANCE if CONDITIONAL else None, 'guidance_strength': GUIDANCE_STRENGTH,
#                  'loss_type': 'hybrid', 'device': device}
#     MODEL_ARGS = {'resolution': 256, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 1, 2, 2, 4, 4),
#                   'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 256,
#                   'num_res_blocks': 2, 'split_qkv_first': False,
#                   'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000 if CONDITIONAL else None}
#     state_dict = convert_state_dict(torch.load(STATE_DICT_FILENAME, map_location="cpu"))
# else:
#     raise NotImplementedError(STATE_DICT_FILENAME)
