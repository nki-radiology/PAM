import torch
import numpy     as np
import torch.nn  as nn
from   collections import OrderedDict

def conv_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.Conv2d
    elif dim_conv == 3:
        return nn.Conv3d
    
    
def conv_gl_avg_pool_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.AdaptiveAvgPool2d
    elif dim_conv == 3:
        return nn.AdaptiveAvgPool3d
 
 
def conv_up_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.ConvTranspose2d
    elif dim_conv == 3:
        return nn.ConvTranspose3d


def up_sample_mode(dim_conv: int):
    if dim_conv   == 2:
        return 'nearest'
    elif dim_conv == 3:
        return 'trilinear'


def max_pool_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.MaxPool2d
    elif dim_conv == 3:
        return nn.MaxPool3d
    

def dim_after_n_layers(size, n_layers):
    for _ in range(n_layers):
        size = np.ceil(size/2)
    return size


class Conv(nn.Module):
    def __init__(self, input_ch, output_ch, group_num, dim):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            conv_layer(dim)(in_channels=input_ch, out_channels=output_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm   (num_groups=group_num, num_channels=output_ch),
            nn.ReLU        (inplace=True),
            conv_layer(dim)(in_channels=output_ch, out_channels=output_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm   (num_groups=group_num,  num_channels=output_ch),
            nn.ReLU        (inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out       



class Up_Conv(nn.Module):

    def __init__(self, input_ch, output_ch, group_num, dim):
        super(Up_Conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample (scale_factor=2,        mode=up_sample_mode(dim)),
            nn.Conv3d   (in_channels=input_ch,  out_channels=output_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups =group_num, num_channels=output_ch),
            nn.ReLU     (inplace=True)
        )

    def forward(self, x):
        out = self.up(x)
        return out
 
 
 
class Encoder(nn.Module):
    def __init__(self,
                input_ch  : int = 1,
                input_dim : int = [192, 192, 160],
                group_num : int = 8,
                filters   : object = [32, 64, 128, 256] 
                ):
        super(Encoder, self).__init__()
        """
        Inputs:
            - input_dim  : Dimensionality of the input 
            - latent_dim : Dimensionality of the latent space (Z)
            - groups     : Number of groups in the normalization layers
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.group_num  = group_num
        self.filters    = filters

        modules = OrderedDict()
        
        for layer_i, layer_filters in enumerate(filters):

            modules['encoder_block_' + str(layer_i)] = nn.Sequential(
                conv_layer(len(input_dim))(
                    in_channels=input_ch, out_channels=layer_filters, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=group_num, num_channels=layer_filters),
                nn.ReLU(inplace=True)
            )
            input_ch = layer_filters
        
        self.conv_net = nn.Sequential(modules)


    def forward(self, x):
        x = self.conv_net(x)
        return x