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

# Experiments and Examples
To demonstrate the model's usefulness and relative ease to train, I trained it with both the 'digits' and 'letters' splits of the [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset). The results are highlighted below. Pretrained models and command line arguments for training and sampling are provided in the next sections.

# Sampling
Command line interface with diff_sample.py is done by creating variables for the sampling and model arguments. For each pre-trained model, I linked a download of the state_dict, a recommended **_$SAMPLE_ARGS_**, and the correct **_$MODEL_ARGS_**. Changing the **_$MODEL_ARGS_** will break stuff. 
Here is a list of changes to the arguments in **_$SAMPLE_ARGS_** that you can make to fit it to your uses:
  - `--batch_size (INT)`: replace (INT) with number of images to make each batch
  - `--num_samples (INT)`: replace (INT) with number of batches to generate. total number of samples made will be batch_size * num_samples
  - `--rescaled_num_steps (INT)`: replace (INT) with number of steps you want each denoising process to take
  - `--use_ddim`: include this to use DDIM sampling (useful for when rescaled_num_steps is small)
  - `--ddim_eta (FLOAT)`: replace (FLOAT) with decimal to control DDIM strength during sampling
  - `--model_path (FILENAME)`: replace (FILENAME) with relative path to pre-trained model state dict
  - `--save_path (DIRECTORY)`: replace (DIRECTORY) with relative path to directory to save samples in
  - `--guidance_method (STRING)`: replace (STRING) with either `classifier` or `classifier_free` to use the corresponding guidance method
  - `--guidance_strength (FLOAT)`: replace (FLOAT) with decimal to control guidance strength during sampling
  - `--upsample`: include this to upsample samples by 4x before displaying/saving (make sure you have the ESRGAN state dict at `models/RealESRGAN_x4plus.pth`
  - `--labels (X/Y/Z/...)`: replace (X/Y/Z/...) with '/'-separated list of labels to use instead of random labels. list must be num_samples long (for example, to create 3 samples with labels 1, 2, and 3 respectively, add `--labels 1/2/3`)
  - `--cpu`: include this to force the sampler to use the cpu instead of autodetecting device. this is useful if encountering cuda memory errors

The correct **_$MODEL_ARGS_** variable is provided under each pre-trained model (and should not be changed), along with a recommended **_$SAMPLE_ARGS_** (which you might wanna change the `--batch_size` and `--num_samples` of and maybe add a `--save_path`). You can sample by cd'ing into `nice-diffusion/` and executing the provided commands with whatever modifications you wish to make (you can drop the `-w` to not print updates during sampling).

The use of upsampling requires that the [RealESRGAN_x4plus.pth](https://download1641.mediafire.com/gpmb5azvul0g/6o6hazgj2h7tlsb/RealESRGAN_x4plus.pth) file is downloaded and located in the `models/` folder.

### EMNIST
**Download pre-trained model**: [EMNIST_model_params.pt](https://download1594.mediafire.com/q3isbeoo7s7g/se37uu47y07us19/EMNIST_model_params.pt "Download EMNIST Model")

The statements written below assume that the pre-trained model is my EMNIST model and that it is called [EMNIST_model_params.pt](https://download1594.mediafire.com/q3isbeoo7s7g/se37uu47y07us19/EMNIST_model_params.pt "Download EMNIST Model") and is located in a folder called `models/`. If it's somewhere else, please fix the `--model_path` argument.
```PowerShell
# Linux Terminal 
SAMPLE_ARGS='--batch_size 8 --num_samples 1 --rescaled_num_steps 25 --model_path models/EMNIST_model_params.pt --guidance_method classifier_free --guidance_strength 0.8'
MODEL_ARGS='--resolution 28 --num_classes 26 --in_channels 1 --model_channels 64 --attention_resolutions 7/14 --channel_mult 1/2/4 --num_res_blocks 2 --num_heads 4 --split_qkv_first --resblock_updown --use_adaptive_gn --beta_schedule cosine --sampling_var_type learned_interpolation'
python diff_sample.py -w $SAMPLE_ARGS $MODEL_ARGS
```
```PowerShell
# PowerShell 
$SAMPLE_ARGS=???--batch_size 8 --num_samples 1 --rescaled_num_steps 25 --model_path models/EMNIST_model_params.pt --guidance_method classifier_free --guidance_strength 0.8???
$MODEL_ARGS=???--resolution 28 --num_classes 26 --in_channels 1 --model_channels 64 --attention_resolutions 7/14 --channel_mult 1/2/4 --num_res_blocks 2 --num_heads 4 --split_qkv_first --resblock_updown --use_adaptive_gn --beta_schedule cosine --sampling_var_type learned_interpolation???
python diff_sample.py -w $SAMPLE_ARGS $MODEL_ARGS
```

### OpenAI 64x64 Conditional ImageNet
**Download pre-trained model**: [64x64_diffusion.pt](https://download1478.mediafire.com/5i0iy57fy7yg/7fbkanlblkjjbpk/64x64_diffusion.pt "Download Converted 64x64 ImageNet Model")

The statements written below assume that the pre-trained model is OpenAI's 64x64 Conditional ImageNet model (Dhariwal/Nichol, Diffusion Model Beats GAN on Image Synthesis, https://arxiv.org/pdf/2105.05233.pdf) and that it is called [64x64_diffusion.pt](https://download1478.mediafire.com/5i0iy57fy7yg/7fbkanlblkjjbpk/64x64_diffusion.pt "Download Converted 64x64 ImageNet Model") and is located in a folder called `models/`. If it's somewhere else, please fix the`--model_path` argument.
```PowerShell
# Linux Terminal 
SAMPLE_ARGS='--batch_size 8 --num_samples 1 --rescaled_num_steps 25 --use_ddim --ddim_eta 0.0 --model_path models/64x64_diffusion.pt'
MODEL_ARGS='--resolution 64 --num_classes 1000 --in_channels 3 --model_channels 192 --attention_resolutions 8/16/32 --channel_mult 1/2/3/4 --num_res_blocks 3 --num_head_channels 64 --split_qkv_first --resblock_updown --use_adaptive_gn --beta_schedule cosine --sampling_var_type learned_interpolation'
python diff_sample.py -w $SAMPLE_ARGS $MODEL_ARGS
```
```PowerShell
# PowerShell 
$SAMPLE_ARGS=???--batch_size 8 --num_samples 1 --rescaled_num_steps 25 --use_ddim --ddim_eta 0.0 --model_path models/64x64_diffusion.pt???
$MODEL_ARGS=???--resolution 64 --num_classes 1000 --in_channels 3 --model_channels 192 --attention_resolutions 8/16/32 --channel_mult 1/2/3/4 --num_res_blocks 3 --num_head_channels 64 --split_qkv_first --resblock_updown --use_adaptive_gn --beta_schedule cosine --sampling_var_type learned_interpolation???
python diff_sample.py -w $SAMPLE_ARGS $MODEL_ARGS
```

### OpenAI 128x128 Conditional ImageNet
**Download pre-trained model**: [128x128_diffusion.pt](https://download1326.mediafire.com/rt93wwag56eg/zl6hqoaywpud94u/128x128_diffusion.pt "Download Converted 128x128 ImageNet Model")

The statements written below assume that the pre-trained model is OpenAI's 128x128 Conditional ImageNet model (Dhariwal/Nichol, Diffusion Model Beats GAN on Image Synthesis, https://arxiv.org/pdf/2105.05233.pdf) and that it is called [128x128_diffusion.pt](https://download1326.mediafire.com/rt93wwag56eg/zl6hqoaywpud94u/128x128_diffusion.pt "Download Converted 128x128 ImageNet Model") and is located in a folder called `models/`. If it's somewhere else, please fix the `--model_path` argument.
```PowerShell
# Linux Terminal 
SAMPLE_ARGS='--batch_size 8 --num_samples 1 --rescaled_num_steps 25 --use_ddim --ddim_eta 0.0 --model_path models/128x128_diffusion.pt'
MODEL_ARGS='--resolution 128 --num_classes 1000 --in_channels 3 --model_channels 256 --attention_resolutions 8/16/32 --channel_mult 1/1/2/3/4 --num_res_blocks 2 --num_heads 4 --resblock_updown --use_adaptive_gn --beta_schedule linear --sampling_var_type learned_interpolation'
python diff_sample.py -w $SAMPLE_ARGS $MODEL_ARGS
```
```PowerShell
# PowerShell 
$SAMPLE_ARGS=???--batch_size 8 --num_samples 1 --rescaled_num_steps 25 --use_ddim --ddim_eta 0.0 --model_path models/128x128_diffusion.pt???
$MODEL_ARGS=???--resolution 128 --num_classes 1000 --in_channels 3 --model_channels 256 --attention_resolutions 8/16/32 --channel_mult 1/1/2/3/4 --num_res_blocks 2 --num_heads 4 --resblock_updown --use_adaptive_gn --beta_schedule linear --sampling_var_type learned_interpolation???
python diff_sample.py -w $SAMPLE_ARGS $MODEL_ARGS
```

### OpenAI 256x256 Unconditional ImageNet
**Download pre-trained model**: [256x256_diffusion_uncond.pt](https://download1347.mediafire.com/5kimx3bn6hcg/8224m8buzgi4zvw/256x256_diffusion_uncond.pt "Download Converted 256x256 Unconditional ImageNet Model")

The statements written below assume that the pre-trained model is OpenAI's 256x256 Unconditional ImageNet model (Dhariwal/Nichol, Diffusion Model Beats GAN on Image Synthesis, https://arxiv.org/pdf/2105.05233.pdf) and that it is called [256x256_diffusion_uncond.pt](https://download1347.mediafire.com/5kimx3bn6hcg/8224m8buzgi4zvw/256x256_diffusion_uncond.pt "Download Converted 256x256 Unconditional ImageNet Model") and is located in a folder called `models/`. If it's somewhere else, please fix the `--model_path` argument.
```PowerShell
# Linux Terminal 
SAMPLE_ARGS='--batch_size 8 --num_samples 1 --rescaled_num_steps 25 --use_ddim --ddim_eta 0.0 --model_path models/256x256_diffusion_uncond.pt'
MODEL_ARGS='--resolution 256 --in_channels 3 --model_channels 256 --attention_resolutions 8/16/32 --channel_mult 1/1/2/2/4/4 --num_res_blocks 2 --num_head_channels 64 --resblock_updown --use_adaptive_gn --beta_schedule linear --sampling_var_type learned_interpolation'
python diff_sample.py -w $SAMPLE_ARGS $MODEL_ARGS
```
```PowerShell
# PowerShell 
$SAMPLE_ARGS=???--batch_size 8 --num_samples 1 --rescaled_num_steps 25 --use_ddim --ddim_eta 0.0 --model_path models/256x256_diffusion_uncond.pt???
$MODEL_ARGS=???--resolution 256 --in_channels 3 --model_channels 256 --attention_resolutions 8/16/32 --channel_mult 1/1/2/2/4/4 --num_res_blocks 2 --num_head_channels 64 --resblock_updown --use_adaptive_gn --beta_schedule linear --sampling_var_type learned_interpolation???
python diff_sample.py -w $SAMPLE_ARGS $MODEL_ARGS
```

# Training
write this
  
## Planned additions:
  - will add script to train a noisy classifier so that classifier-based sampling guidance can actually be used
  - will add distributed training to allow cloud training
  - will wrap into a complete pip-able package
  - smarter timestep sampling during training, a la (Dhariwal/Nichol, Diffusion Model Beats GAN on Image Synthesis, https://arxiv.org/pdf/2105.05233.pdf)
  - may add some sampling tricks like truncated or double sampling during denoising
  - annealed LR during training?
  - smarter respacing (more steps in beginning, etc)
