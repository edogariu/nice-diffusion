## nice-diffusion
# PyTorch implementation of Gaussian Diffusion model.
Improvements over the seminal Diffusion model (Ho et al., Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2006.11239.pdf) are listed below, 
along with the papers they come from:
  - Rescaled denoising process for fewer steps when sampling: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - Learned variances in addition to epslions: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - KL-divergence, NLL, and hybrid loss functions: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - Classifier-based sampling guidance: (Dhariwal/Nichol, Improved Denoising Diffusion Probabilistic Models, https://arxiv.org/pdf/2102.09672.pdf)
  - Classifier-free sampling guidance: (Ho/Salismans, Classifier-Free Diffusion Guidance, https://openreview.net/pdf/ea628d03c92a49b54bc2d757d209e024e7885980.pdf)
  - DDIM sampling: (Song et al., Denoising Diffusion Implicit Models, https://arxiv.org/pdf/2010.02502.pdf)
  - Upscaling after sampling: (Wang et al., Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data, https://arxiv.org/pdf/2107.10833.pdf)
  
  
Planned improvements:
  - will add script to train a noisy classifier so that classifier-based guidance can actually be used
  - smarter timestep sampling during training, a la (Dhariwal/Nichol Diffusion Model Beats GAN on Image Synthesis (OpenAI): https://arxiv.org/pdf/2105.05233.pdf)
  - may add some sampling tricks like truncated or double sampling
