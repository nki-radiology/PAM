import numpy as np
import torch as nn
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
