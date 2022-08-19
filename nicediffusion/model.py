from abc import abstractmethod

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

import torch.utils
import torch.utils.checkpoint

from .utils import checkpoint, override

'''
DIFFUSION MODEL BASED ON THE FOLLOWING PAPERS AND CORRESPONDING WORK:
    - Ho et al. Denoising Diffusion Probabilistic Models (DDPM): https://arxiv.org/pdf/2006.11239.pdf
    - Song et al. Denoising Diffusion Implicit Models (DDIM): https://arxiv.org/pdf/2010.02502.pdf
    - Dhariwal/Nichol Diffusion Model Beats GAN on Image Synthesis (OpenAI): https://arxiv.org/pdf/2105.05233.pdf
    
    Model is a UNet architecture (https://arxiv.org/pdf/1505.04597.pdf). From OpenAI paper:
    "The UNet model uses a stack of residual layers and downsampling convolutions, followed by a stack of 
    residual layers with upsampling convolutions, with skip connections connecting the layers with the same 
    spatial size." and
    "In the rest of the paper, we use this final improved model architecture as our default: variable width with 2 
    residual blocks per resolution, multiple heads with 64 channels per head, attention at 32, 16 and 8 resolutions, 
    BigGAN residual blocks for up and downsampling, and adaptive group normalization for injecting timestep and 
    class embeddings into residual blocks."
'''


# Abstract wrapper class to be used when forward propagation needs step embeddings
class UsesSteps(nn.Module):
    @abstractmethod
    def forward(self, x, step):
        """
        To be used when forward propagation needs step embeddings
        """


# Wrapper sequential class that knows when to pass time step embedding to its children or not
class UsesStepsSequential(nn.Sequential, UsesSteps):
    @override
    def forward(self, x, step):
        for layer in self:
            if isinstance(layer, UsesSteps):
                x = layer(x, step)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Creates a module whose forward pass performs 2x upsampling of input image.

    Parameters
    --------------
        - in_channels (int): number of channels to pass in
        - out_channels (int): number of channels to return. if None, return with in_channels channels
        - with_conv (bool)L if True use Conv2D for upsampling, if False use nearest neighbor interpolation


    Returns
    ---------------
        - An nn.Module to be used to compose a network.
    """

    def __init__(self, in_channels, with_conv, out_channels=None):
        super(Upsample, self).__init__()
        self.with_conv = with_conv

        if with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels if out_channels is not None else in_channels,
                                  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Creates a module whose forward pass performs 2x downsampling of input image.

    Parameters
    ---------------
        - in_channels (int): number of channels to pass in
        - out_channels (int): number of channels to return. if None, return with in_channels channels
        - with_conv (bool): if True use Conv2D for downsampling, if False use 2D avg. pooling

    Returns
    --------------
        - An nn.Module to be used to compose a network.
    """

    def __init__(self, in_channels, with_conv, out_channels=None):
        super(Downsample, self).__init__()
        self.with_conv = with_conv

        if with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels if out_channels is not None else in_channels,
                                  kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # ADD ASYMMETRIC PADDING??

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        return x


# Can be used with up or downsampling
# If use_conv=True and out_channels!=None, use spatial convolution to change number of channels
class ResidualBlock(UsesSteps):
    """
    Creates a residual block containing 2 convolutional layers with a skip connection.
    Uses timestep embeddings and 2 layers of GroupNorm (32 groups).

    Parameters
    --------------
        - in_channels (int): number of channels to pass in
        - out_channels (int): number of channels to return. if None, return with in_channels channels
        - step_channels (int): number of channels to use for timestep embedding
        - upsample (bool): whether to upsample before first convolution
        - downsample (bool): whether to downsample before first convolution
        - use_conv (bool): if out_channels != in_channels and this is True, use 3x3 convolution to
        - change number of channels
        - use_adaptive_gn (bool): whether to use step embedding to scale and shift input or simply add them
        - use_grad_checkpoints (bool): whether to use grad checkpointing, which lowers memory but raises computation
        - dropout (double): dropout probability

    Returns
    ------------------
        - An nn.Module to be used to compose a network.
    """

    def __init__(self, in_channels, step_channels, dropout, upsample=False, downsample=False,
                 use_conv=False, out_channels=None, use_adaptive_gn=False, use_grad_checkpoints=False):
        super(ResidualBlock, self).__init__()

        self.use_conv = use_conv
        self.use_adaptive_gn = use_adaptive_gn
        self.silu = nn.SiLU()
        out_channels = out_channels if out_channels is not None else in_channels
        self.use_grad_checkpoints = use_grad_checkpoints

        if upsample:
            self.h_resample = Upsample(in_channels=in_channels, with_conv=False)
            self.x_resample = Upsample(in_channels=in_channels, with_conv=False)
            self.resample = True
        elif downsample:
            self.h_resample = Downsample(in_channels=in_channels, with_conv=False)
            self.x_resample = Downsample(in_channels=in_channels, with_conv=False)
            self.resample = True
        else:
            self.h_resample = self.x_resample = nn.Identity()
            self.resample = False

        # Change number of channels in skip connection if necessary
        if out_channels == in_channels:
            self.skip = nn.Identity()
        elif use_conv:
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(1, 1), stride=(1, 1))

        self.in_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-5)
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-5)
        self.out_conv = zero_module(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        if use_adaptive_gn:
            self.step_embedding = nn.Linear(step_channels, 2 * out_channels)
        else:
            self.step_embedding = nn.Linear(step_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, step):
        return checkpoint(self._forward, (x, step), self.parameters(), self.use_grad_checkpoints)

    def _forward(self, x, step):
        h = x
        h = self.silu(self.in_norm(h))
        if self.resample:
            h = self.h_resample(h)
            x = self.x_resample(x)
        h = self.in_conv(h)

        # add in timestep embedding
        embedding = self.step_embedding(self.silu(step))[:, :, None, None]
        # apply Adaptive GroupNorm (scale-shift norm) from (sec. 3 of OpenAI paper)
        if self.use_adaptive_gn:
            # [y_s, y_b] = y
            scale, shift = torch.chunk(embedding, 2, dim=1)
            # AdaGN(h, y) = y_s * GroupNorm(h) + y_b
            h = self.out_norm(h) * (1 + scale) + shift
        else:
            h += embedding
            h = self.out_norm(h)
        h = self.silu(h)
        h = self.dropout(h)
        h = self.out_conv(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Creates an attention-style Transformer block containing QKV-attention, a residual connection,
    and a linear projection out.
    Uses 1 layer of GroupNorm (32 groups).

    Parameters
    -------------
        - channels (int): number of channels to pass in and return out
        - num_heads (int): number of heads to use in Multi-Headed-Attention
        - num_head_channels (int): number of channels to use for each head. if not None, then this block uses
        (channels // num_head_channels) heads and ignores num_heads
        - split_qkv_first (bool): whether to split qkv first or split heads first during attention

    Returns
    -------------
        - An nn.Module to be used to compose a network.
    """

    def __init__(self, channels, num_heads=1, num_head_channels=None, split_qkv_first=True):
        super(AttentionBlock, self).__init__()

        if num_head_channels is None:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), "channels {} is not divisible by num_head_channels {}".format(channels, num_head_channels)
            self.num_heads = channels // num_head_channels

        self.split_qkv_first = split_qkv_first
        self.scale = (channels // self.num_heads) ** -0.5

        self.qkv_nin = nn.Conv1d(in_channels=channels, out_channels=3 * channels,
                                 kernel_size=(1,), stride=(1,))

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-5)
        # self.proj_out = zero_module(nn.Conv1d(in_channels=channels, out_channels=channels,
        #                                       kernel_size=(1,), stride=(1,)))
        self.proj_out = zero_module(nn.Conv1d(in_channels=channels, out_channels=channels,
                                              kernel_size=(1,), stride=(1,)))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)  # always checkpoint on attention blocks

    # My implementation of MHA
    def _forward(self, x):
        # compute 2-D attention (condense H,W dims into N)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        qkv = self.norm(x)

        if self.split_qkv_first:
            # Get q, k, v
            qkv = self.qkv_nin(qkv).permute(0, 2, 1)  # b,c,hw -> b,hw,c
            qkv = qkv.reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # q/k/v.shape = b,num_heads,hw,c//num_heads

            # w = softmax(q @ k / sqrt(d_k))
            w = (q @ k.transpose(2, 3)) * self.scale
            w = torch.softmax(w, dim=-1)

            # attention = w @ v
            h = (w @ v).transpose(1, 2).reshape(B, H * W, C).permute(0, 2, 1)
        else:
            qkv = self.qkv_nin(qkv)  # b, 3 * c, hw
            q, k, v = qkv.reshape(B * self.num_heads, 3 * C // self.num_heads, H * W).split(C // self.num_heads, dim=1)

            # w = softmax(q @ k / sqrt(d_k))
            w = q.transpose(1, 2) @ k * self.scale
            w = torch.softmax(w, dim=2)

            # h = w @ v
            h = torch.einsum("bts,bcs->bct", w, v).reshape(B, -1, H * W)

        h = self.proj_out(h)

        return (h + x).reshape(B, C, H, W)


class DiffusionModel(nn.Module):
    """
    Creates a Diffusion model to predict epsilon in a generative denoising process.

    Parameters
    ------------
        - resolution (int): height and width resolution of inputs (assumes square images)
        - in_channels (int): number of channels to pass in
        - out_channels (int): number of channels to return
        - model_channels (int): number of channels to use within the model, before any channel multipliers apply
        - channel_mult (tuple of ints): multipliers for number of inner channels to use
        - num_res_blocks (int): number of ResidualBlocks to use for each channel multiplier level
        - resblock_updown (bool): whether to use ResidualBlocks or Upsample/Downsample modules to resample
        - conv_resample (bool): if resblock_updown is False, whether to use Conv2D layers to resample
        - attention_resolutions (tuple of ints): which resolutions at which to apply attention
        - num_classes (int): if not None, number of classes to use for class-conditional models
        - num_heads (int): number of heads to use for AttentionBlocks
        - num_head_channels (int): number of channels to use for each head for AttentionBlocks, supersedes num_heads
        - use_adaptive_gn (bool): whether to use Adaptive GroupNorm with step & class embeddings in ResidualBlocks
        - split_qkv_first (bool): whether to split qkv first or split heads first during attention
        - use_grad_checkpoints (bool): whether to use grad checkpointing, which lowers memory but raises computation
        - dropout (double): dropout probability in the ResidualBlocks

    Returns
    --------------
    - A UNet model to be used for diffusion.
    """

    def __init__(
            self,
            resolution,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            num_classes=None,
            num_heads=1,
            num_head_channels=None,
            resblock_updown=False,
            use_adaptive_gn=False,
            split_qkv_first=True,
            use_grad_checkpoints=False
    ):
        super(DiffusionModel, self).__init__()
        self.resolution = resolution
        self.in_channels = in_channels

        # Create embedding pipeline for time steps
        step_embed_dim = 4 * model_channels
        self.model_channels = model_channels
        self.step_embed = nn.Sequential(
            nn.Linear(model_channels, step_embed_dim),
            nn.SiLU(),
            nn.Linear(step_embed_dim, step_embed_dim)
        )

        # Create embedding for classes
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, embedding_dim=step_embed_dim)
            self.conditional = True
            self.num_classes = num_classes
        else:
            self.conditional = False
            self.num_classes = None

        # Downsampling blocks
        self._feature_size = curr_channels = input_channels = int(model_channels * channel_mult[0])
        curr_res = resolution

        self.downsampling = nn.ModuleList([UsesStepsSequential(torch.nn.Conv2d(in_channels, curr_channels,
                                                                               kernel_size=(3, 3), stride=(1, 1),
                                                                               padding=(1, 1)))])
        input_block_channels = [curr_channels]  # Keep track of how many channels each block operates with
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # num_res_blocks residual blocks per level, plus an attention layer if specified
                layers = [ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim, dropout=dropout,
                                        out_channels=int(model_channels * mult),
                                        use_adaptive_gn=use_adaptive_gn, use_grad_checkpoints=use_grad_checkpoints)]
                curr_channels = int(model_channels * mult)

                # Add attention layer if specified to be added at this downsampling
                if curr_res in attention_resolutions:
                    layers.append(AttentionBlock(channels=curr_channels, split_qkv_first=split_qkv_first,
                                                 num_heads=num_heads, num_head_channels=num_head_channels))
                input_block_channels.append(curr_channels)
                self.downsampling.append(UsesStepsSequential(*layers))
                self._feature_size += curr_channels

            curr_channels = int(model_channels * mult)
            # Downsample at levels where the channel is multiplied
            if level != len(channel_mult) - 1:
                output_channels = curr_channels
                if resblock_updown:
                    self.downsampling.append(UsesStepsSequential(
                        ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim, dropout=dropout,
                                      out_channels=output_channels, downsample=True,
                                      use_adaptive_gn=use_adaptive_gn, use_grad_checkpoints=use_grad_checkpoints)))
                else:
                    self.downsampling.append(UsesStepsSequential(
                        Downsample(in_channels=curr_channels, out_channels=output_channels, with_conv=conv_resample)))
                curr_channels = output_channels
                input_block_channels.append(curr_channels)
                curr_res //= 2
                self._feature_size += curr_channels

        # Middle blocks - residual, attention, residual
        layers = [ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim, dropout=dropout,
                                use_adaptive_gn=use_adaptive_gn, use_grad_checkpoints=use_grad_checkpoints),
                  AttentionBlock(channels=curr_channels, split_qkv_first=split_qkv_first,
                                 num_heads=num_heads, num_head_channels=num_head_channels),
                  ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim, dropout=dropout,
                                use_adaptive_gn=use_adaptive_gn, use_grad_checkpoints=use_grad_checkpoints)]
        self.middle_block = UsesStepsSequential(*layers)
        self._feature_size += curr_channels

        # Upsampling blocks (pretty much reverse of downsampling blocks, also includes skip connections from before
        # various downsampling layers)
        self.upsampling = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [ResidualBlock(in_channels=curr_channels + skip_channels,  # append for skip connections
                                        step_channels=step_embed_dim, use_adaptive_gn=use_adaptive_gn,
                                        dropout=dropout, out_channels=int(model_channels * mult),
                                        use_grad_checkpoints=use_grad_checkpoints)]
                curr_channels = int(model_channels * mult)
                if curr_res in attention_resolutions:
                    layers.append(AttentionBlock(channels=curr_channels, split_qkv_first=split_qkv_first,
                                                 num_heads=num_heads, num_head_channels=num_head_channels))

                # Upsample at each channel multiplier layer
                if level != 0 and i == num_res_blocks:
                    output_channels = curr_channels
                    if resblock_updown:
                        layers.append(ResidualBlock(in_channels=curr_channels, step_channels=step_embed_dim,
                                                    dropout=dropout, out_channels=output_channels,
                                                    upsample=True, use_adaptive_gn=use_adaptive_gn,
                                                    use_grad_checkpoints=use_grad_checkpoints))
                    else:
                        layers.append(Upsample(in_channels=curr_channels, out_channels=output_channels,
                                               with_conv=conv_resample))
                    curr_res *= 2

                self._feature_size += curr_channels
                self.upsampling.append(UsesStepsSequential(*layers))

        # Output
        self.out = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=curr_channels),
                                 nn.SiLU(),
                                 zero_module(nn.Conv2d(in_channels=input_channels, out_channels=out_channels,
                                                       kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))

    def forward(self, x, timestep, y=None):
        assert (y is not None) == self.conditional, 'pass y iff class-conditional model'
        assert x.shape[2] == self.resolution and x.shape[3] == self.resolution, \
            'incorrect resolution: {}'.format(x.shape[2:])

        embedding = self.step_embed(timestep_embedding(timestep, self.model_channels))

        if self.conditional:
            embedding += self.class_embedding(y)

        # input and downsampling
        xs = []
        for module in self.downsampling:
            x = module(x, embedding)
            xs.append(x)

        # middle block (res, attn, res)
        x = self.middle_block(x, embedding)

        # output and upsampling
        for module in self.upsampling:
            # Concatenate for skip connections from before the downsampling layers
            if not isinstance(module, Upsample):
                x = torch.cat([x, xs.pop()], dim=1)
            x = module(x, embedding)
        return self.out(x)


class SuperResolutionModel(DiffusionModel):
    """
    Diffusion model to perform Super Resolution of low-res image. Takes as input the low_res image in the forward pass.

        Parameters:
            - upscale_resolution (int): the resolution to upscale to
            - in_channels (int): the number of channels in the input image. usually 3
            - all of DiffusionModel's params

            - low_res (torch.tensor): low-resolution image to upscale
    """

    def __init__(self, upscale_resolution, in_channels, **args):
        super().__init__(upscale_resolution, in_channels * 2, **args)

    def forward(self, x, timesteps, low_res=None, y=None):
        assert low_res is not None, 'must pass low resolution image to SuperResolutionModel.forward'
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear", align_corners=False)
        x = torch.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, y)


# ---------------------------------------------------------------------------------------------------------------------
# HELPER METHODS FOR MODEL IMPLEMENTATION
# ---------------------------------------------------------------------------------------------------------------------

# Helpful method that zeros all the parameters in a nn.Module, used for initialization
def zero_module(module):
    for param in module.parameters():
        param.detach().zero_()
    return module


# Method to create sinusoidal timestep embeddings, much like positional encodings found in Transformers
def timestep_embedding(timesteps, embedding_dim, max_period=10000):
    half = embedding_dim // 2
    emb = math.log(max_period) / half
    emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb).to(timesteps.device)
    emb = timesteps[:, None].float() * emb[None]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
    # Zero pad for odd dimensions
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb
