import torch.nn  as nn
import numpy     as np
from   collections import OrderedDict
    
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


def dim_after_n_layers(size, n_layers):
    for _ in range(n_layers):
        size = np.ceil(size/2)
    return size


class Encoder(nn.Module):
    def __init__(self,
                input_ch  : int = 1,
                input_dim : int = [256, 256, 512],
                latent_dim: int = 512,
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
        self.latent_dim = latent_dim
        self.group_num  = group_num
        self.filters    = filters 

        modules = OrderedDict()
        
        for layer_i, layer_filters in enumerate(filters):

            modules['encoder_block_' + str(layer_i)] = nn.Sequential(
                conv_layer(len(input_dim))(
                    in_channels=input_ch, out_channels=layer_filters, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=group_num, num_channels=layer_filters),
                nn.ReLU()
            )
            input_ch = layer_filters
        
        self.conv_net = nn.Sequential(modules)

        output_dim = [dim_after_n_layers(i, layer_i+1) for i in input_dim]
        self.elem  = int(layer_filters * np.prod(output_dim))
            
        self.fc_mu  = nn.Linear(in_features=self.elem, out_features=latent_dim, bias=False)
        self.fc_var = nn.Linear(in_features=self.elem, out_features=latent_dim, bias=False)

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, self.elem)
        mu, sigma = self.fc_mu(x), self.fc_var(x)
        return mu, sigma
    
    
    
class Decoder(nn.Module):
    def __init__(self,
                 output_ch : int = 1,
                 input_dim : int = [256, 256, 512],
                 latent_dim: int = 512,
                 group_num : int = 8, 
                 filters   : object = [32, 64, 128, 256]):
        super(Decoder, self).__init__()
        """
        Inputs:
            - input_ch   : Number of input channels of the image. For medical images, this parameter usually is 1
            - latent_dim : Dimensionality of the latent space (Z)
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """        
        self.output_ch  = output_ch
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.group_num  = group_num
        self.filters    = filters 

        modules = OrderedDict()

        filters             = filters[::-1]
        self.last_feature   = filters[0]
        input_dec           = [dim_after_n_layers(i, len(filters)) for i in input_dim]
        self.input_decoder  = list(map(int, input_dec))
        elem                = int(filters[0] * np.prod(input_dec))

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=elem)
        )

        for layer_i in range(len(filters) - 1):
            modules['decoder_block_' + str(layer_i)] = nn.Sequential(
                conv_up_layer(len(input_dim))(
                    in_channels=filters[layer_i], out_channels=filters[layer_i+1], kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=group_num, num_channels=filters[layer_i+1]),
                nn.ReLU()
            )
        self.conv_up_net = nn.Sequential(modules)

        self.final_layer = nn.Sequential(
                conv_up_layer(len(input_dim))(
                    in_channels=filters[layer_i+1], out_channels=output_ch, kernel_size=4, stride=2, padding=1, bias=False)
        )
        
    
    def forward(self, x):
        print('Shape input: ', x.shape)
        x = self.input_layer(x)
        print('Shape: ', x.shape)

        if len(self.input_decoder) == 2:
            x = x.view(-1, self.last_feature, self.input_decoder[0], self.input_decoder[1])
        elif len(self.input_decoder) == 3:
            x = x.view(-1, self.last_feature, self.input_decoder[0], self.input_decoder[1], self.input_decoder[2])
        else:
            NotImplementedError('only support 2d and 3d')
        print('Shape: ', x.shape)
        x = self.conv_up_net(x)
        print('Shape: ', x.shape)
        x = self.final_layer(x)
        print('Shape End: ', x.shape)

        return x


class Encoder_WAE(nn.Module):
    def __init__(self,
                input_ch  : int = 1,
                input_dim : int = [192, 192, 304],
                latent_dim: int = 512,
                group_num : int = 8,
                filters   : object = [32, 64, 128, 256] 
                ):
        super(Encoder_WAE, self).__init__()
        """
        Inputs:
            - input_dim  : Dimensionality of the input 
            - latent_dim : Dimensionality of the latent space (Z)
            - groups     : Number of groups in the normalization layers
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.group_num  = group_num
        self.filters    = filters

        modules = OrderedDict()
        
        for layer_i, layer_filters in enumerate(filters):

            modules['encoder_block_' + str(layer_i)] = nn.Sequential(
                conv_layer(len(input_dim))(
                    in_channels=input_ch, out_channels=layer_filters, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=group_num, num_channels=layer_filters),
                nn.ReLU()
            )
            input_ch = layer_filters
        
        self.conv_net = nn.Sequential(modules)

        output_dim = [dim_after_n_layers(i, layer_i+1) for i in input_dim]
        self.elem  = int(layer_filters * np.prod(output_dim))
            
        self.latent_space_z = nn.Sequential(
            nn.Linear(in_features=self.elem, out_features=latent_dim, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_net(x)
        print(' Encoder shape 1: ', x.shape)
        x = x.view(-1, self.elem)
        print('Encoder shape: ', x.shape)
        latent_space_z = self.latent_space_z(x)
        return latent_space_z


class Encoder_Discriminator(nn.Module):
    def __init__(self,
                input_ch  : int = 1,
                input_dim : int = [192, 192, 304],
                group_num : int = 8,
                filters   : object = [32, 64, 128, 256] 
                ):
        super(Encoder_Discriminator, self).__init__()
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
                nn.ReLU()
            )
            input_ch = layer_filters
        
        self.conv_net = nn.Sequential(modules)


    def forward(self, x):
        x = self.conv_net(x)
        return x
    