## nice-diffusion
# PyTorch implementation of Generative Gaussian Diffusion Model
This is my implementation of an advanced, improved diffusion model combining different techniques from various research papers over the past couple years. I did this with the intention of applying it for painting generation and artist style transfer (see https://www.github.com/edogariu/artist-style-transfer), but beacuse of computing limitations (AWS if ur reading this please get back to me lol) I have yet to make it work. However, I tried to make it as general to use for any generative setting as possible, and am working on developing it into a standalone package.

## Improvements over DDPM
Implemented improvements over the seminal Diffusion model (Ho et al., Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2006.11239.pdf) are listed below, along with the papers they come from:
  - Rescaled denoising process for fewer steps when sampling: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - Learned variances in addition to learned epsilons: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - KL-divergence, NLL, and hybrid loss functions: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - Classifier-based sampling guidance: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - Classifier-free sampling guidance: (Ho/Salismans, Classifier-Free Diffusion Guidance, https://openreview.net/pdf/ea628d03c92a49b54bc2d757d209e024e7885980.pdf)
  - DDIM sampling: (Song et al., Denoising Diffusion Implicit Models, https://arxiv.org/pdf/2010.02502.pdf)
  - Upscaling after sampling: (Wang et al., Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data, https://arxiv.org/pdf/2107.10833.pdf)

# How to Use
To install the package, clone the repository (duh), cd into `nice-diffusion/`, and execute `pip3 install .`. The rest just works. If you don't wish to upsample ever, you will be able to save installation time by commenting out the `basicsr` requirement from `setup.py`. Once you have the package, you can simply import it elsewhere. Two scripts are located in `scripts/` to show how to use the package, and are described below.

## Sampling
To sample a model, you first need to download the state dictionary for that model. For each pre-trained model, I linked a download of the state_dict below. The path to the corresponding file must be passed as a command line argument of the form `--model_path [PATH]`.

Here is a list of other arguments to pass when sampling one of these pre-trained models:
   - `--model_path [FILENAME] (required)`: replace [FILENAME] with relative path to pre-trained model state dict
  - `--batch_size [INT] (required)`: replace [INT] with number of images to make each batch
  - `--num_samples [INT] (required)`: replace [INT] with number of batches to generate. total number of samples made will be batch_size * num_samples
  - `-w (optional)`: include this to print wordy updates of the sampling progress. defaults to silence.
  - `--save_path [DIRECTORY] (optional)`: replace [DIRECTORY] with relative path to directory to save samples in. defaults to displaying output
  - `--upsample (optional)`: include this to upsample samples by 4x before displaying/saving (make sure you have the ESRGAN state dict at `models/RealESRGAN_x4plus.pth`). defaults to no upsampling
  - `--labels [X/Y/Z/...] (optional)`: replace [X/Y/Z/...] with '/'-separated list of labels to use. list must be num_samples long (for example, to create 3 samples with labels 1, 2, and 3 respectively, add `--labels 1/2/3`). defaults to random labels
  - `--cpu (optional)`: include this to force the sampler to use the cpu. this is useful if encountering cuda memory errors. defaults to autodetecting device

The use of upsampling requires that the [RealESRGAN_x4plus.pth](https://download1641.mediafire.com/gpmb5azvul0g/6o6hazgj2h7tlsb/RealESRGAN_x4plus.pth) file is downloaded and located in the `models/` folder.

### Pre-Trained Models
  - EMNIST
    - **Download pre-trained model**: [EMNIST_model_params.pt](https://download1594.mediafire.com/q3isbeoo7s7g/se37uu47y07us19/EMNIST_model_params.pt "Download EMNIST Model")
    - To demonstrate the model's usefulness and relative ease to train, I trained it with the 'letters' split of the [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset).
  - OpenAI 64x64 Conditional ImageNet
    - **Download pre-trained model**: [64x64_diffusion.pt](https://download1478.mediafire.com/5i0iy57fy7yg/7fbkanlblkjjbpk/64x64_diffusion.pt "Download Converted 64x64 ImageNet Model") 
    - I assume that the pre-trained model is OpenAI's 64x64 Conditional ImageNet model (Dhariwal/Nichol, Diffusion Model Beats GAN on Image Synthesis, https://arxiv.org/pdf/2105.05233.pdf) and that it is called [64x64_diffusion.pt](https://download1478.mediafire.com/5i0iy57fy7yg/7fbkanlblkjjbpk/64x64_diffusion.pt "Download Converted 64x64 ImageNet Model"). 
  - OpenAI 128x128 Conditional ImageNet
    - **Download pre-trained model**: [128x128_diffusion.pt](https://download1326.mediafire.com/rt93wwag56eg/zl6hqoaywpud94u/128x128_diffusion.pt "Download Converted 128x128 ImageNet Model")
    - I assume that the pre-trained model is OpenAI's 128x128 Conditional ImageNet model (Dhariwal/Nichol, Diffusion Model Beats GAN on Image Synthesis, https://arxiv.org/pdf/2105.05233.pdf) and that it is called [128x128_diffusion.pt](https://download1326.mediafire.com/rt93wwag56eg/zl6hqoaywpud94u/128x128_diffusion.pt "Download Converted 128x128 ImageNet Model"). 
  - OpenAI 256x256 Unconditional ImageNet
    - **Download pre-trained model**: [256x256_diffusion_uncond.pt](https://download1347.mediafire.com/5kimx3bn6hcg/8224m8buzgi4zvw/256x256_diffusion_uncond.pt "Download Converted 256x256 Unconditional ImageNet Model")
    - I assume that the pre-trained model is OpenAI's 256x256 Unconditional ImageNet model (Dhariwal/Nichol, Diffusion Model Beats GAN on Image Synthesis, https://arxiv.org/pdf/2105.05233.pdf) and that it is called [256x256_diffusion_uncond.pt](https://download1347.mediafire.com/5kimx3bn6hcg/8224m8buzgi4zvw/256x256_diffusion_uncond.pt "Download Converted 256x256 Unconditional ImageNet Model").

## Training
Take a look at the training script in `scripts/train.py`. I haven't yet put together nice command line arguments for this, but you just create a `nicediffusion.model.DiffusionModel` object and a `nicediffusion.trainer.Trainer` object and supply the right parameters. Training a model requires designing all the configurations for model architecture and diffusion parameters; for help with this, look at the parameters for `nicediffusion.model.DiffusionModel` and `nicediffusion.diffusion.Diffusion`, as well as the ones for the pre-trained models in `nicediffusion.default_args`. If you wish to sample from a customly-trained model, you can either add it to the list of default arguments and edit the argparser in `nicediffusion.utils`, which will make passing only the model path behave nicely, or you can pass in the `--custom` flag to `scripts/sample.py` along with all the necessary parameter arguments (this is tedious). 

Just get out there and have fun :)
  
## Planned additions:
  - add patience, validation, etc. for Trainer (basically just clean it up)
  - add proper evaluation metrics
  - will set up cmd line args for train script
  - will add stuff to train a noisy classifier so that classifier-based sampling guidance can actually be used
  - will add distributed training to allow cloud training
  - smarter timestep sampling during training, a la (Dhariwal/Nichol, Diffusion Model Beats GAN on Image Synthesis, https://arxiv.org/pdf/2105.05233.pdf)
  - may add some sampling tricks like truncated or double sampling during denoising
  - annealed LR during training?
  - smarter respacing (more steps in beginning, etc)
