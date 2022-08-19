# -------------------------------------------
# ------------- FOR EMNIST MODEL ------------
# -------------------------------------------
EMNIST_DIFFUSION_ARGS = {'rescaled_num_steps': 25, 'original_num_steps': 1000, 'use_ddim': False, 'ddim_eta': 0.0,
                    'beta_schedule': 'cosine', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                    'guidance_method': 'classifier_free', 'guidance_strength': 0.8, 'loss_type': 'hybrid'}
EMNIST_MODEL_ARGS = {'resolution': 28, 'attention_resolutions': (7, 14), 'channel_mult': (1, 2, 4),
                'num_heads': 4, 'in_channels': 1, 'out_channels': 2, 'model_channels': 64,
                'num_res_blocks': 2, 'split_qkv_first': True, 'dropout': 0.05,
                'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 27}

# -------------------------------------------
# ------------- FOR 64x64 MODEL -------------
# -------------------------------------------
OPENAI_64_DIFFUSION_ARGS = {'rescaled_num_steps': 25, 'original_num_steps': 1000, 'use_ddim': True, 'ddim_eta': 0.0,
                    'beta_schedule': 'cosine', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                    'guidance_method': None, 'guidance_strength': 0.8, 'loss_type': 'hybrid'}
OPENAI_64_MODEL_ARGS = {'resolution': 64, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 2, 3, 4),
                'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 192,
                'num_res_blocks': 3, 'split_qkv_first': True, 'dropout': 0.05,
                'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000}

# -------------------------------------------
# ------------- FOR 128x128 MODEL -----------
# -------------------------------------------
OPENAI_128_DIFFUSION_ARGS = {'rescaled_num_steps': 25, 'original_num_steps': 1000, 'use_ddim': True, 'ddim_eta': 0.0,
                    'beta_schedule': 'linear', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                    'guidance_method': None, 'guidance_strength': 0.8, 'loss_type': 'hybrid'}
OPENAI_128_MODEL_ARGS = {'resolution': 128, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 1, 2, 3, 4),
                'num_heads': 4, 'in_channels': 3, 'out_channels': 6, 'model_channels': 256,
                'num_res_blocks': 2, 'split_qkv_first': True, 'dropout': 0.05,
                'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000}

# -------------------------------------------
# ------------- FOR 256x256 MODEL -----------
# -------------------------------------------
OPENAI_256_DIFFUSION_ARGS = {'rescaled_num_steps': 25, 'original_num_steps': 1000, 'use_ddim': True, 'ddim_eta': 0.0,
                    'beta_schedule': 'linear', 'sampling_var_type': 'learned_interpolation', 'classifier': None,
                    'guidance_method': None, 'guidance_strength': 0.8, 'loss_type': 'hybrid'}
OPENAI_256_MODEL_ARGS = {'resolution': 256, 'attention_resolutions': (8, 16, 32), 'channel_mult': (1, 1, 2, 2, 4, 4),
                'num_head_channels': 64, 'in_channels': 3, 'out_channels': 6, 'model_channels': 256,
                'num_res_blocks': 2, 'split_qkv_first': True, 'dropout': 0.05,
                'resblock_updown': True, 'use_adaptive_gn': True, 'num_classes': 1000}
    