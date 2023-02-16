import torch.nn  as nn

    
def conv_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.Conv2d
    elif dim_conv == 3:
        return nn.Conv3d


def conv_up_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.ConvTranspose2d
    elif dim_conv == 3:
        return nn.ConvTranspose3d


def conv_gl_avg_pool_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.AdaptiveAvgPool2d
    elif dim_conv == 3:
        return nn.AdaptiveAvgPool3d


def get_conv_layer(type_conv   : str, 
                   dim         : int,
                   in_channels : int, 
                   out_channels: int, 
                   kernel_size : int, 
                   stride      : int, 
                   padding     : int):
    if type_conv == 'Up':
        return conv_up_layer(dim)( in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding )