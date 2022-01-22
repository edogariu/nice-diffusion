from collections import OrderedDict
import argparse
import matplotlib.pyplot as plt
import numpy as np

def make_argparser(prog):
    """
    Creates argparser for specified program.
    """
    if prog == 'diff_sample':
        description = 'Sample images generated from Diffusion Model.'
        is_sample = True  # flag to keep track of what we are using the parser for
    elif prog == 'diff_train':
        description = 'Train Diffusion Model.'
        is_sample = False
    else:
        raise NotImplementedError(prog)
    o = '(optional)'
    r = '(required)'
    parser = argparse.ArgumentParser(prog=prog, description=description)

    # Sampling-only args
    if is_sample:
        sampling = parser.add_argument_group('sampling arguments', 'arguments for sampling process')
        sampling.add_argument('--model_path', type=str, required=True, metavar=r,
                              help='relative file path of model state dict')
        sampling.add_argument('--batch_size', type=int, required=True, metavar=r,
                              help='number of images per batch. decrease this to conserve cuda memory')
        sampling.add_argument('--num_samples', type=int, required=True, metavar=r,
                              help='number of batches to sample. total number of images is num_samples * batch_size')
        sampling.add_argument('--upsample', required=False, default=False, action='store_true',
                              help='add to use Real-ESRGAN (https://arxiv.org/abs/2107.10833) for '
                                   '4x superresolution')
        sampling.add_argument('--wordy', '--w', dest='wordy', required=False, default=False, action='store_true',
                              help='add this to print status')
        sampling.add_argument('--save_path', type=str, required=False, metavar=o, default=None,
                              help='relative file path to save generated images. if not provided, they will be '
                                   'displayed instead')
        sampling.add_argument('--labels', type=str, required=False, metavar=o, default='',
                              help='labels to be split among samples during conditional sampling. '
                                   'if not provided, labels will be random')
        sampling.add_argument('--start_img', type=str, required=False, metavar=o, default=None,
                              help='relative file path of image to start denoising with. if not provided, start from '
                                   'random noise')
        sampling.add_argument('--steps_to_do', type=int, required=False, metavar=o, default=None,
                              help='number of rescaled steps of noise to apply to start_img and '
                                   'then unapply via denoising')
        sampling.add_argument('--seed', type=int, required=False, metavar=o, default=None,
                              help='rng seed to use for reproducibility')

    # Training-only args
    else:
        training = parser.add_argument_group('training arguments', 'arguments for training process')
        training.add_argument('--batch_size', type=int, required=True, metavar=r,
                              help='number of images per batch. decrease this to conserve cuda memory')
        training.add_argument('--lr', type=float, required=True, metavar=r,
                              help='learning rate to use for optimizer')
        training.add_argument('--weight_decay', type=float, required=True, metavar=r,
                              help='weight decay to use for optimizer')
        training.add_argument('--iterations', type=int, required=True, metavar=r,
                              help='number of iterations to train for')
        training.add_argument('--resume_step', type=int, required=False, metavar=o, default=0,
                              help='step to resume training at. loads correct checkpoint from /checkpoints/ folder')
        training.add_argument('--wordy', '--w', dest='wordy', required=False, default=False, action='store_true',
                              help='add this to print status')
        training.add_argument('--save_every', type=int, required=False, metavar=o, default=None,
                              help='how often to save checkpoint in /checkpoints/ folder. defaults to not saving')
        training.add_argument('--sample_every', type=int, required=False, metavar=o, default=None,
                              help='how often to sample 8 images from model during training. defaults to not sampling')
        training.add_argument('--ema_rate', type=float, required=False, metavar=o, default=0.9999,
                              help='rate to perform exponential moving average with')
        training.add_argument('--use_fp16', required=False, default=False, action='store_true',
                              help='add this to train with fp_16 for improved time and memory performance')
        training.add_argument('--grad_accumulation', type=int, required=False, metavar=o, default=1,
                              help='how often to do optimizer step (gradient accumulation). defaults to 1')
        training.add_argument('--seed', type=int, required=False, metavar=o, default=None,
                              help='rng seed to use for reproducibility')

    # DiffusionModel args
    model_args = parser.add_argument_group('model arguments', 'arguments to create DiffusionModel')
    model_args.add_argument('--resolution', type=int, required=True, metavar=r,
                            help='resolution of images to generate')
    model_args.add_argument('--model_channels', type=int, required=True, metavar=r,
                            help='number of model channels to start with')
    model_args.add_argument('--channel_mult', type=str, required=True, metavar=r,
                            help='model channel multipliers')
    model_args.add_argument('--num_res_blocks', type=int, required=True, metavar=r,
                            help='number of residual blocks to use in each channel mult layer')
    model_args.add_argument('--attention_resolutions', type=str, required=True, metavar=r,
                            help='which resolutions to apply attention')
    model_args.add_argument('--num_classes', type=int, required=False, default=None, metavar=o,
                            help='number of classes to use for conditional model. defaults to unconditional model')
    model_args.add_argument('--dropout', type=float, required=not is_sample, default=0.0, metavar=o if is_sample else r,
                            help='dropout probability')
    model_args.add_argument('--in_channels', type=int, required=False, default=3, metavar=o,
                            help='number of channels to input to model. defaults to 3')
    model_args.add_argument('--num_heads', type=int, required=False, default=4, metavar=o,
                            help='number of heads to use in multi-headed-attention. defaults to 4')
    model_args.add_argument('--num_head_channels', type=int, required=False, default=None, metavar=o,
                            help='number of channels for each head in multi-headed-attention. overrides num_heads')
    model_args.add_argument('--split_qkv_first', required=False, default=False, action='store_true',
                            help='add this to use a different attention order')
    model_args.add_argument('--resblock_updown', required=False, default=False, action='store_true',
                            help='add this to use residual blocks for up/downsampling instead of interpolation/pooling')
    model_args.add_argument('--use_adaptive_gn', required=False, default=False, action='store_true',
                            help='add this to use adaptive GroupNorm')

    # Diffusion args
    diff_args = parser.add_argument_group('diffusion arguments', 'arguments to determine diffusion/denoising process')
    diff_args.add_argument('--rescaled_num_steps', type=int, required=True, metavar=r,
                           help='number of diffusion timesteps to rescale setup to use')
    diff_args.add_argument('--beta_schedule', type=str, required=True, metavar=r,
                           help='schedule to use for noise variances. can be \'linear\', \'cosine\', or \'constant\'')
    diff_args.add_argument('--sampling_var_type', type=str, required=True, metavar=r,
                           help='method to use for learning sampling variances. can be \'small\', \'large\', '
                                '\'learned\', or \'learned_interpolation\'')
    diff_args.add_argument('--use_ddim', required=False, default=False, action='store_true',
                           help='add this to use DDIM sampling')
    diff_args.add_argument('--ddim_eta', type=float, required=False, default=0.0, metavar=o,
                           help='eta value to use for DDIM sampling')
    diff_args.add_argument('--original_num_steps', type=int, required=False, default=1000, metavar=o,
                           help='number of diffusion timesteps for setup to originally use. defaults to 1000')
    diff_args.add_argument('--loss_type', type=str, required=not is_sample, default='hybrid',
                           metavar=o if is_sample else r,
                           help='method to calculate loss. can be \'simple\', \'KL\', \'KL_rescaled\', or \'hybrid\'')
    diff_args.add_argument('--guidance_method', type=str, required=False, default=None, metavar=o,
                           help='guidance method to use during sampling. can be \'classifier\' or \'classifier_free\'')
    diff_args.add_argument('--guidance_strength', type=float, required=False, default=None, metavar=o,
                           help='strength of applied sampling guidance')
    diff_args.add_argument('--classifier_path', metavar=o, type=str, required=False, default=None,
                           help='relative file path of classifier state dict to use for classifier guidance')
    return parser


def get_dicts_from_args(args):
    """
    Creates model_args and diffusion_args dicts to be passed into constructors for DiffusionModel and Diffusion, as well
    as an extra dict for all other hyperparameters, such as number of samples or learning rate, etc.
    """
    model_keys = ['resolution', 'attention_resolutions', 'channel_mult', 'num_heads', 'in_channels', 'out_channels',
                  'model_channels', 'num_res_blocks', 'split_qkv_first', 'dropout',
                  'resblock_updown', 'use_adaptive_gn', 'num_classes']
    diff_keys = ['rescaled_num_steps', 'original_num_steps', 'use_ddim', 'ddim_eta', 'beta_schedule',
                 'sampling_var_type', 'classifier', 'guidance_method', 'guidance_strength', 'loss_type']
    args = vars(args)
    model_args = {}
    diff_args = {}
    other_args = {}
    for key, val in args.items():
        if key in model_keys:
            model_args[key] = val
        elif key in diff_keys:
            diff_args[key] = val
        else:
            other_args[key] = val

    assert (diff_args['guidance_method'] == 'classifier_free' or diff_args['guidance_method'] == 'classifier') == \
           (model_args['num_classes'] is not None), 'use guidance only for conditional models'
    assert (diff_args['guidance_method'] == 'classifier') == (other_args['classifier_path'] is not None)
    if other_args['classifier_path'] is not None:
        raise NotImplementedError('i have not yet implemented a noisy classifier, sorry')

    # Split strings for attention resolutions, channel multipliers, and labels
    if other_args.__contains__('labels') and len(other_args['labels']) > 0:
        other_args['labels'] = [int(i) for i in other_args['labels'].split('/')]
    model_args['attention_resolutions'] = [int(i) for i in model_args['attention_resolutions'].split('/')]
    model_args['channel_mult'] = [int(i) for i in model_args['channel_mult'].split('/')]

    # Use double out_channels if we are also learning variances
    if diff_args['sampling_var_type'] == 'learned' or diff_args['sampling_var_type'] == 'learned_interpolation':
        model_args['out_channels'] = model_args['in_channels'] * 2
    else:
        model_args['out_channels'] = model_args['in_channels']

    # Use an extra class if classifier_free guidance
    if diff_args['guidance_method'] == 'classifier_free':
        model_args['num_classes'] += 1

    return other_args, model_args, diff_args


def convert_state_dict(sd):
    """
    Convert state dict from the ones from github.com/openai/guided-diffusion to one compatible with my model.
    Does not change contents of input state_dict, returns new one.
    """
    def convert_param_name(name):
        name = name.replace('input_blocks', 'downsampling')
        name = name.replace('output_blocks', 'upsampling')
        name = name.replace('in_layers.0', 'in_norm')
        name = name.replace('in_layers.2', 'in_conv')
        name = name.replace('emb_layers.1', 'step_embedding')
        name = name.replace('out_layers.0', 'out_norm')
        name = name.replace('out_layers.3', 'out_conv')
        name = name.replace('skip_connection', 'skip')
        name = name.replace('time_embed', 'step_embed')
        name = name.replace('qkv', 'qkv_nin')
        name = name.replace('label_emb', 'class_embedding')
        return name

    new_sd = OrderedDict()
    for _ in range(len(sd)):
        key, val = sd.popitem(False)
        old_key = key
        key = convert_param_name(key)
        sd[old_key] = val
        new_sd[key] = val

    return new_sd

# Show image (img must be RGB and from [0.0, 1.0] or [0, 255])
def imshow(img, title=None, colormap=None):
    plt.imshow(img.astype(np.uint8), cmap=colormap)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def grab_checkpoint(step):
    """
    Returns tuple of paths representing a checkpoint from given training step.
    """
    return ('checkpoints/{}_model_params.pt'.format(step), 'checkpoints/{}_ema_params.pt'.format(step),
            'checkpoints/{}_opt_params.pt'.format(step), step)


class Rescale(object):
    """
    Rescale tensor image from [0, 1] to [-1, 1].
    """
    def __call__(self, im):
        return 2 * im - 1


def cycle(dataloader):
    """
    Changes dataloader from something to be iterated through to something to be cycled through with next().
    """
    while True:
        for data in dataloader:
            yield data

def override(a):  # silly function i had to write to write @override, sometimes python can be annoying lol
    return a
