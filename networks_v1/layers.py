import torch
import numpy    as np
import torch.nn as nn
from   torch.nn import ReLU, LeakyReLU, GroupNorm


def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):

    """ autocrop
    Helper function that allows to crop the encoder layers if their sizes do not match with the corresponding layers
    from the decoder block.

    :param encoder_layer: Features from the encoder block
    :type encoder_layer: torch.Tensor
    :param decoder_layer: Features from the decoder block
    :type decoder_layer: torch.Tensor

    :return: It returns the cropped layers in case they have different sizes, otherwise it returns the original layers
    :rtype: torch.Tensor
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
    """ convolutional layer
    Function to apply a convolution layer considering a dimension, eg. Conv2d or Conv3d.

    :param dim: Layer dimension
    :type dim: int

    :return: A convolutional layer
    :rtype: nn.Module
    """
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
    """ get convolutional layer
    Returns a 2D or 3D convolutional layer according to the assigned parameters.
    In case there is no specification of values in the parameters, the default values are considered.

    :param in_channels: Number of channels-features in the input image
    :type in_channels: int
    :param out_channels: Number of channels-features produced by the convolution
    :type out_channels: int
    :param kernel_size: Size of the convolving kernel. Default: 3
    :type kernel_size: int
    :param stride: Stride of the convolution. Default: 1
    :type  stride: int
    :param padding: Padding added to all sides of the input. Default: 1
    :type padding: int
    :param bias: If true, it adds a learnable bias to the output. Default: True
    :type bias: bool
    :param dim: A convolution dimension
    :type dim: int

    :return: A convolutional layer
    :rtype: nn.Module
    """
    return conv_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)



def conv_transpose_layer(dim: int):
    """ transposed convolutional layer
    Function to apply a transposed convolution layer considering a dimension, eg. ConvTranspose2d or ConvTranspose3d.

    :param dim: Layer dimension
    :type dim: int

    :return: A transposed convolutional layer
    :rtype: nn.Module
    """
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

    """ get transposed convolutional layer
    Returns a 2D or 3D transposed convolutional layer according to the assigned parameters.
    In case there is no specification of values in the parameters, the default values are considered.

    :param in_channels: Number of channels-features in the input image
    :type in_channels: int
    :param out_channels: Number of channels-features produced by the convolution
    :type out_channels: int
    :param kernel_size: Size of the convolving kernel. Default: 2
    :type kernel_size: int
    :param stride: Stride of the convolution. Default: 2
    :type stride: int
    :param dim: Dimension of the transposed convolution
    :type dim: int
    :param up_mode: If 'tranposed', it applies a transposed convolution, otherwise, it applies an Upsamle.
                    Default: 'tranposed'
    :type up_mode: str

    :return: A transposed or upsampled convolutional layer
    :rtype: nn.Module
    """
    if up_mode == 'transposed':
        return conv_transpose_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    else:
        return nn.Upsample(scale_factor=2.0, mode=up_mode)



def maxpool_layer(dim: int):
    """ max pooling layer
    Function to apply a max pooling over an input considering a dimension, eg. MaxPool2d or MaxPool3d.

    :param dim: Layer dimension
    :type dim: int

    :return: A max pooling layer
    :rtype: nn.Module
    """
    if dim   == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d


def get_maxpool_layer(kernel_size   : int = 2,
                      stride        : int = 2,
                      padding       : int = 0,
                      dim           : int = 2):
    """ get max pooling layer
    Returns a 2D or 3D max pooling layer according to the assigned parameters.
    In case there is no specification of values in the parameters, the default values are considered.

    :param kernel_size: Size of the window to take a max over. Default: 2
    :type kernel_size: int
    :param stride: Stride of the window. Default: 2
    :type stride: int
    :param padding: Padding added to all sides. Default: 0
    :type padding: int
    :param dim: Layer Dimension. Default : 2
    :type dim: int

    :return: A max pooling layer layer
    :rtype: nn.Module
    """
    return maxpool_layer(dim=dim)(kernel_size=kernel_size, stride=stride, padding=padding)


def global_avg_pool_layer(dim: int):
    """ global average pooling layer
    Function to apply a global average pooling over an input considering a dimension,
    e.g. AdaptiveAvgPool2d or AdaptiveAvgPool2d.

    :param dim: Layer dimension
    :type dim: int

    :return: A global average pooling layer
    :rtype: nn.Module
    """
    if dim   == 3:
        return nn.AdaptiveAvgPool3d
    elif dim == 2:
        return nn.AdaptiveAvgPool2d


def get_global_avg_pool_layer(output_size: tuple, dim: int):
    """ get global average layer
    Returns a 2D or 3D global average pooling layer according to the assigned parameters.
    In case there is no specification of values in the parameters, the default values are considered.

    :param output_size: Number of output size
    :type output_size: tuple
    :param dim: Layer dimension
    :type dim: int

    :return: A global average pooling layer
    :rtype: nn.Module
    """
    if dim == 3:
        avg_pool_layer = nn.Sequential(
            #nn.ReLU(),
            nn.AdaptiveAvgPool3d(output_size),
            nn.ReLU()
        )
    else:
        avg_pool_layer = nn.Sequential(
            #nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size),
            nn.ReLU()
        )
    return avg_pool_layer

    #return global_avg_pool_layer(dim=dim)(output_size)


def get_activation(activation: str):
    """ get activation function
    Returns an activation function. This function can return one of the three activation function:
    nn.ReLU, nn.LeakyReLU or nn.ELU.

    :param activation: activation function name
    :type activation: str

    :return: A type of activation function
    :rtype: nn.Module
    """
    if activation   == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()


def get_normalization(normalization : str,
                      num_channels  : int,
                      dim           : int):
    """ get normalization
    Returns a 2D or 3D normalization according to the assigned size and dimension. this function can return one of three
    types of normalization: nnBatchNorm, nn.InstanceNorm, nn.GroupNorm.

    :param normalization: Type of normalization, eg. 'batch', 'instance' or 'group'
    :type normalization: str
    :param num_channels: Number of features
    :type num_channels: int
    :param dim: Input dimension
    :type dim: int

    :return: A type of normalization
    :rtype: nn.Module
    """
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



def weight_init(module, method, **kwargs):
    """ weights initialization
    Initialize the weights of a module.

    :param module: modules to initialize
    :type module: nn.Module
    :param method: method to use in the initialization.
    :type method: nn.Module

    :return: weights initialization
    :rtype: nn.Module
    """
    if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
        method(module.weight, **kwargs)  # weights
        #print("Initialization Weights")


def bias_init(module, method, **kwargs):
    """ bias initialization
    Initialize the bias of a module.

    :param module: modules to initialize
    :type module: nn.Module
    :param method: method to use in the initialization. Default: zeros_
    :type method: nn.Module

    :return: bias initialization
    :rtype: nn.Module
    """
    if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
        method(module.bias, **kwargs)   # bias
        #print("Initialization Bias")



def initialize_parameters(modules,
                          method_weights= nn.init.kaiming_uniform_,
                          method_bias   = nn.init.zeros_,
                          kwargs_weights= {},
                          kwargs_bias   = {}
                          ):
    """ parameters initialization
    Initialize the bias and weights of a module.

    :param modules: modules to initialize
    :type modules: nn.Module
    :param method_weights: method to initialize the weights. Default: kaiming_uniform_
    :type method_weights: nn.Module
    :param method_bias: method to initialize the bias. Default: zeros_
    :type method_bias: nn.Module

    :return: bias initialization
    :rtype: nn.Module
    """
    for module in modules:
        weight_init(module, method_weights, **kwargs_weights)  # initialize weights
        bias_init(module, method_bias, **kwargs_bias)          # initialize bias