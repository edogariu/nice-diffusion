## nice-diffusion
# Custom PyTorch implementation of Generative Gaussian Diffusion Model
This is an implementation of a state-of-the-art diffusion model combining different improvements and techniques from various research papers over the past year or two. I did this with the intention of applying it for painting generation and artist style transfer (see https://www.github.com/edogariu/artist-style-transfer), but beacuse of computing limitations (AWS if ur reading this please get back to me lol) I have yet to make it work. However, I tried to make it as general to use for any generative setting (conditional or not), and am working on developing it into a standalone package.

## Improvements over DDPM
Implemented improvements over the seminal Diffusion model (Ho et al., Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2006.11239.pdf) are listed below, along with the papers they come from:
  - Rescaled denoising process for fewer steps when sampling: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - Learned variances in addition to learned epsilons: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - KL-divergence, NLL, and hybrid loss functions: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - Classifier-based sampling guidance: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - Classifier-free sampling guidance: (Ho/Salismans, Classifier-Free Diffusion Guidance, https://openreview.net/pdf/ea628d03c92a49b54bc2d757d209e024e7885980.pdf)
  - DDIM sampling: (Song et al., Denoising Diffusion Implicit Models, https://arxiv.org/pdf/2010.02502.pdf)
  - Upscaling after sampling: (Wang et al., Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data, https://arxiv.org/pdf/2107.10833.pdf)

## Experiments and Examples
To demonstrate the model's usefulness and relative ease to train, I trained it with both the 'digits' and 'letters' splits of the EMNIST dataset (https://www.nist.gov/itl/products-and-services/emnist-dataset). The results are highlighted below. Pretrained models are available in the '/checkpoints/' or '/models/' folder, and command line arguments for training and sampling are provided below.

## Sampling
For sampling, first ensure that you have a pre-trained model downloaded somewhere, as well as a folder called '/samples/' if you wish to save generated samples and a folder called '/models/' containing the state dict 'RealESRGAN_x4plus.pth' if you wish to upsample. 

The command line statements written below assume that the pre-trained model is my EMNIST model and that it is called '20000_model_params.pt' and is located in a folder called '/checkpoints/'.
for evan:
  - add saving to diff_sample, and make upsample work for 1-channel images too
  - add scripts for sampling from other pretrained models

## Training
write this
  
## Planned additions:
  - will add script to train a noisy classifier so that classifier-based sampling guidance can actually be used
  - will add distributed training to allow cloud training
  - will add gradient checkpointing to conserve memory
  - will wrap into a complete pip-able package
  - smarter timestep sampling during training, a la (Dhariwal/Nichol, Diffusion Model Beats GAN on Image Synthesis, https://arxiv.org/pdf/2105.05233.pdf)
  - may add some sampling tricks like truncated or double sampling during denoising
  - annealed LR during training?
