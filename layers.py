import numpy as np
import torch
import torch.nn as nn
from   torch.nn import ReLU, LeakyReLU, GroupNorm

"""
Convolution functions for 2D and 3D data
 - conv               : to get the proper convolution dimension
 - convolution        : to apply a convolution operator over an input
 - encoding_relu      : encoding block using ReLU as activation function
 - encoding_leaky_relu: encoding block using LeakyReLU as activation function
"""


def conv(dim=2):
    if dim == 2:
        return nn.Conv2d
    return nn.Conv3d


def convolution(in_ch, out_ch, kernel_size, stride, dim=2):
    return conv(dim=dim)(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=1)


# PAM encoding block
def encoding_relu(in_ch, out_ch, kernel_size, stride, dim=2, num_groups=4):
    encoding = nn.Sequential(
        convolution(in_ch, out_ch, kernel_size, stride, dim=dim),
        GroupNorm(num_groups, out_ch),
        ReLU()
    )
    return encoding


def encoding_leaky_relu(in_ch, out_ch, kernel_size, stride, dim=2, num_groups=4):
    encoding = nn.Sequential(
        convolution(in_ch, out_ch, kernel_size, stride, dim=dim),
        GroupNorm(num_groups, out_ch),
        LeakyReLU(0.1)
    )
    return encoding


"""
Deconvolution functions for 2D and 3D data
 - de_conv            : to get the proper deconvolution dimension
 - deconvolution      : to apply a deconvolution operator over an input
 - decoding_relu      : decoding block using ReLU as activation function
 - decoding_leaky_relu: decoding block using LeakyReLU as activation function
"""


def de_conv(dim=2):
    if dim == 2:
        return nn.ConvTranspose2d
    return nn.ConvTranspose3d


def deconvolution(in_ch, out_ch, kernel_size, stride, dim=2):
    return de_conv(dim=dim)(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=1)


# PAM decoding block
def decoding_relu(in_ch, out_ch, kernel_size, stride, dim=2, num_groups=4):
    decoding = nn.Sequential(
        deconvolution(in_ch, out_ch, kernel_size, stride, dim=dim),
        GroupNorm(num_groups, out_ch),
        ReLU()
    )
    return decoding


def decoding_leaky_relu(in_ch, out_ch, kernel_size, stride, dim=2, num_groups=4):
    decoding = nn.Sequential(
        deconvolution(in_ch, out_ch, kernel_size, stride, dim=dim),
        GroupNorm(num_groups, out_ch),
        LeakyReLU(0.1)
    )
    return decoding


"""
Weights Initialization
"""


def he_init(model):
    if isinstance(model, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(model.weight, mode="fan_out", nonlinearity="relu")

    if isinstance(model, nn.Linear):
        size = model.weight.size()
        fan_out = size[0]
        fan_in  = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        model.weight.data.normal_(0.0, variance)

    elif isinstance(model, GroupNorm):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)

# ----------------------------------------------------------------------------------------------------------------------


@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer


def conv_layer(dim: int):
    if dim   == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d


def get_conv_layer(in_channels  : int,
                   out_channels : int,
                   kernel_size  : int  = 3,
                   stride       : int  = 1,
                   padding      : int  = 1,
                   bias         : bool = True,
                   dim          : int  = 2):
    return conv_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)


def conv_transpose_layer(dim: int):
    if dim   == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d


def get_up_layer(in_channels    : int,
                 out_channels   : int,
                 kernel_size    : int = 2,
                 stride         : int = 2,
                 dim            : int = 3,
                 up_mode        : str = 'transposed',
                 ):
    if up_mode == 'transposed':
        return conv_transpose_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    else:
        return nn.Upsample(scale_factor=2.0, mode=up_mode)


def maxpool_layer(dim: int):
    if dim   == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d


def get_maxpool_layer(kernel_size   : int = 2,
                      stride        : int = 2,
                      padding       : int = 0,
                      dim           : int = 2):
    return maxpool_layer(dim=dim)(kernel_size=kernel_size, stride=stride, padding=padding)


def get_activation(activation: str):
    if activation   == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()


def get_normalization(normalization : str,
                      num_channels  : int,
                      dim           : int):
    if normalization == 'batch':
        if dim   == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        if dim   == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1])  # get the group size from string
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


# ----------------------------------------------------------------------------------------------------------------------
