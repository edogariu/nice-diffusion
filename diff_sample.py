import sys

from cv2 import imread, resize
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import make_argparser, get_dicts_from_args, imshow
from diff_model import DiffusionModel
from diffusion import Diffusion

def main():
    # Parse command lisne arguments
    for _ in range(len(sys.argv)):
        temp = sys.argv.pop(0)s
        for arg in temp.split(' '):
            sys.argv.append(arg)
    parser = make_argparser('diff_sample')
    args = parser.parse_args()
    other_args, model_args, diff_args = get_dicts_from_args(args)

    # Gather hyperparameters
    if other_args['cpu']:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if other_args['seed'] is not None:
        torch.manual_seed(other_args['seed'])
    WORDY = other_args['wordy']
    NUM_SAMPLES = other_args['num_samples']
    BATCH_SIZE = other_args['batch_size']
    UPSAMPLE = other_args['upsample']
    START_IMG, STEPS_TO_DO = other_args['start_img'], other_args['steps_to_do']
    LABELS = other_args['labels']
    CONDITIONAL = (model_args['num_classes'] is not None)
    SAVE_PATH = other_args['save_path']

    model = DiffusionModel(**model_args)
    model.load_state_dict(torch.load(other_args['model_path'], map_location='cpu'), strict=True)
    model.to(device).eval()

    if WORDY:
        print('Model made from {} with {} parameters! :)'.
              format(other_args['model_path'], sum(p.numel() for p in model.parameters())))

        print('Starting Diffusion! There are {} samples of {} images each'.format(NUM_SAMPLES, BATCH_SIZE))
    samples = []
    diffusion = Diffusion(model=model, **diff_args, device=device)

    if START_IMG is not None and STEPS_TO_DO is not None:
        START_IMG = imread(START_IMG)
        # may want to replace this resize with a center crop?
        START_IMG = resize(START_IMG, dsize=(model_args['resolution'], model_args['resolution'])) / 127.5 - 1
        START_IMG = torch.from_numpy(START_IMG).permute(2, 0, 1)[[2, 1, 0], :, :]
        start_imgs = torch.zeros(size=(BATCH_SIZE, model_args['in_channels'],
                                       model_args['resolution'], model_args['resolution']))
        for i_batch in range(BATCH_SIZE):
            temp = START_IMG.clone()
            start_imgs[i_batch] = temp
        START_IMG = start_imgs.to(device)

    if CONDITIONAL and len(LABELS) != 0:
        assert len(LABELS) == NUM_SAMPLES, 'please provide NUM_SAMPLES={} labels'.format(NUM_SAMPLES)
        
    with torch.no_grad():
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
            out = ((out + 1) * 127.5).clamp(0, 255).cpu()
            data = ((data + 1) * 127.5).clamp(0, 255).cpu()
    
            # Convert grayscale to 3-channel images for upsampling, if needed
            if model_args['in_channels'] == 1:
                data = torch.stack((255 - data.squeeze(),) * 3, dim=1)
                out = torch.stack((255 - out.squeeze(),) * 3, dim=1)
    
            if START_IMG is not None:
                START_IMG = ((START_IMG + 1) * 127.5).clamp(0, 255)
                samples.append((START_IMG.cpu(), out, labels.cpu()))
            else:
                samples.append((data, out, labels.cpu() if labels is not None else labels))

    if WORDY:
        if SAVE_PATH is None:
            print('Displaying {} generated images!'.format(NUM_SAMPLES * BATCH_SIZE))
        else:
            print('Saving {} generated images to \'{}\'!'.format(NUM_SAMPLES * BATCH_SIZE, SAVE_PATH))
    if UPSAMPLE:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # lazy import because this is not a common package to have

        if WORDY:
            print('Upsampling to {}x{} resolution!'.format(model_args['resolution'] * 4, model_args['resolution'] * 4))
        model.to(torch.device('cpu'))  # deallocate diffusion model memory
        del model

        # avoid cuda alloc error on my 6GB GPU
        if other_args['cpu'] or (model_args['resolution'] > 64 and BATCH_SIZE > 1) or BATCH_SIZE > 4 or \
                not torch.cuda.is_available():
            upsampling_device = torch.device('cpu')
        else:
            upsampling_device = torch.device('cuda')

        with torch.no_grad():
            esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            esrgan.load_state_dict(torch.load('models/RealESRGAN_x4plus.pth', map_location=upsampling_device)
                                   ['params_ema'], strict=True)
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

    if SAVE_PATH is None:  # Display
        for sample in samples:
            data, out, labels = sample
            # Convert from NCHW-RGB to HWC-RGB
            data = data.permute(0, 2, 3, 1).detach().numpy()
            out = out.to(torch.uint8).permute(0, 2, 3, 1).detach().numpy()
            for b in range(BATCH_SIZE):
                plt.close('all')
                fig = plt.figure(figsize=(7, 3))
                fig.add_subplot(1, 2, 1)
                imshow(data[b], title='Denoising Input')
                fig.add_subplot(1, 2, 2)
                if labels is not None:
                    imshow(out[b], title='Output Image, Label={}'.format(labels[b].detach().item()))
                else:
                    imshow(out[b], title='Output Image')
                plt.waitforbuttonpress()
    else:  # Save
        if model_args['num_classes'] is not None:  # for conditional
            counts = np.zeros(shape=(model_args['num_classes'],), dtype=int)
        else:  # for unconditional
            counts = 0
        for sample in samples:
            _, out, labels = sample
            # Convert from NCHW-RGB to HWC-RGB
            out = out.to(torch.uint8).permute(0, 2, 3, 1).detach().numpy()
            if model_args['in_channels'] == 1:  # Convert 3-channel grayscale images back to 1 channel, if needed
                out = 255 - out[..., 0]
            for b in range(BATCH_SIZE):
                if labels is not None:
                    label = labels[b].detach().item()
                    filename = '{}_sample{}.jpg'.format(label, counts[label])
                    counts[label] += 1
                else:
                    filename = 'sample{}.jpg'.format(counts)
                    counts += 1
                plt.imsave(SAVE_PATH + filename, out[b])

    if WORDY:
        print('Done! have a nice day')
    pass


if __name__ == '__main__':
    main()
