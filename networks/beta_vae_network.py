import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from   torch.autograd       import Variable
from   collections          import OrderedDict
from   networks.layer                import conv_layer
from   networks.layer                import conv_up_layer
from   networks.layer                import conv_gl_avg_pool_layer
from   networks.spatial_transformer  import SpatialTransformer


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu
   
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
        modules = OrderedDict()
        
        for layer_i, layer_filters in enumerate(filters):

            modules['encoder_block' + str(layer_i)] = nn.Sequential(
                conv_layer(len(input_dim))(
                    in_channels=input_ch, out_channels=layer_filters, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=group_num, num_channels=layer_filters),
                nn.GELU()
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

        modules = OrderedDict()

        filters    = filters[::-1]
        self.last_feature = filters[0]
        input_dec  = [dim_after_n_layers(i, len(filters)) for i in input_dim]
        self.input_decoder  = list(map(int, input_dec))
        elem  = int(filters[0] * np.prod(input_dec))

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=elem)
        )

        for layer_i in range(len(filters) - 1):
            modules['decoder_block' + str(layer_i)] = nn.Sequential(
                conv_up_layer(len(input_dim))(
                    in_channels=filters[layer_i], out_channels=filters[layer_i+1], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.GroupNorm(num_groups=group_num, num_channels=filters[layer_i+1]),
                nn.GELU()
            )
        self.conv_up_net = nn.Sequential(modules)

        self.final_layer = nn.Sequential(
                conv_up_layer(len(input_dim))(
                    in_channels=filters[layer_i+1], out_channels=output_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        )
        
    
    def forward(self, x):
        x = self.input_layer(x)

        if len(self.input_decoder) == 2:
            x = x.view(-1, self.last_feature, self.input_decoder[0], self.input_decoder[1])
        elif len(self.input_decoder) == 3:
            x = x.view(-1, self.last_feature, self.input_decoder[0], self.input_decoder[1], self.input_decoder[2])
        else:
            NotImplementedError('only support 2d and 3d')
        x = self.conv_up_net(x)
        x = self.final_layer(x)
        
        return x
     


class Beta_VAE(nn.Module):

    def __init__(self, 
                 input_ch  : int = 1,
                 input_dim : int = [256, 256, 512],
                 latent_dim: int = 512,
                 output_ch : int = 3,
                 filters   : object = [32, 64, 128, 256],
                 group_num : int = 8
                 ):
        super(Beta_VAE, self).__init__()
        """
        Inputs:
            - num_input_channels: Number of input channels of the image. For medical images, this parameter usually is 1
            - z_latent_dim      : Dimensionality of the latent space (Z)
        """     
        # Encoder Block
        self.encoder  = Encoder(input_ch=input_ch, input_dim=input_dim, latent_dim=latent_dim, group_num=group_num, filters=filters)
                         
        # Decoder Block
        self.decoder  = Decoder(output_ch=output_ch, input_dim=input_dim, latent_dim=latent_dim, group_num=group_num, filters=filters)
        
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z              = reparametrize(mu, log_var)      
        x_hat          = self.decoder(z)
        x_hat          = x_hat.view(x.size())
        return x_hat, mu, log_var



class Affine_Beta_VAE(nn.Module):
    def __init__(self,
                 input_ch  : int,
                 data_dim  : int,
                 latent_dim: int,
                 group_num : int,
                 img_shape : object = (256, 256),
                 filters   : object = [16, 32, 64, 128, 256]):
        super(Affine_Beta_VAE, self).__init__()
        """
        Inputs:
            - input_ch   : Number of input channels of the image. For medical images, this parameter usually is 1
            - latent_dim : Dimensionality of the latent space (Z)
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """          
        self.input_ch   = input_ch
        self.data_dim   = data_dim
        self.latent_dim = latent_dim
        self.group_num  = group_num
        self.image_shape= img_shape
        self.filters    = filters
        
        # Encoder Block
        self.encoder_net =  nn.Sequential(OrderedDict([
            ('affine_conv_16'    , conv_layer(self.data_dim)(in_channels=self.input_ch, out_channels=self.filters[0], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)), 
            ('affine_gnorm_16'   , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[0])),
            ('affine_act_fn_16'  , nn.GELU()), # 256 x 256 => 128 x 128
            
            ('affine_conv_32'    , conv_layer(self.data_dim)(in_channels=self.filters[0], out_channels=self.filters[1], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('affine_gnorm_32'   , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[1])),
            ('affine_act_fn_32'  , nn.GELU()), # 128 x 128 => 64 x 64
            
            ('affine_conv_64'    , conv_layer(self.data_dim)(in_channels=self.filters[1], out_channels=self.filters[2], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('affine_gnorm_64'   , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[2])),
            ('affine_act_fn_64'  , nn.GELU()), # 64 x 64 => 32 x 32
            
            ('affine_conv_128'   , conv_layer(self.data_dim)(in_channels=self.filters[2], out_channels=self.filters[3], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('affine_gnorm_128'  , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[3])),
            ('affine_act_fn_128' , nn.GELU()), # 32 x 32 => 16 x 16
            
            ('affine_conv_256'   , conv_layer(self.data_dim)(in_channels=self.filters[3], out_channels=self.filters[4], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('affine_gnorm_256'  , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[4])),
            ('affine_act_fn_256' , nn.GELU()), # 16 x 16 => 8 x 8
        ]))
        
        self.last_linear = nn.Sequential(OrderedDict([
            ('affine_gl_avg_pool' , conv_gl_avg_pool_layer(self.data_dim)(output_size=1)),
            ('affine_ft_vec_all'  , nn.Flatten()),
            ('affine_last_linear' , nn.Linear(in_features=self.filters[4], out_features=1024, bias=False)),
            ('affine_last__act_fn', nn.GELU()), 
        ]))
        
        # Affine Transformation Blocks
        self.dense_w = nn.Sequential(OrderedDict([
            ('affine_w_matrix'       , nn.Linear(in_features=1024, out_features=self.data_dim**2, bias=False)), 
        ]))
        
        self.dense_b = nn.Sequential(OrderedDict([
            ('affine_b_vector'       , nn.Linear(in_features=1024, out_features=self.data_dim, bias=False)), 
        ]))
        
        # Spatial Transformer
        self.spatial_transformer = SpatialTransformer(self.image_shape)
    
    
    def forward(self, fixed: torch.tensor, moving: torch.tensor):
        # Concatenate fixed and moving images
        x = torch.cat((fixed, moving), 1)
        
        # Encoding Block
        x = self.encoder_net(x)
        x = self.last_linear(x)
        
        # Get the degrees of freedom of the affine transformation
        W = self.dense_w(x).view(-1, self.data_dim, self.data_dim)
        b = self.dense_b(x).view(-1, self.data_dim)      
        I = torch.eye(self.data_dim, dtype=torch.float32, device='cuda')
        A = W + I
        
        # Input for the Spatial Transformer Network
        transformation = torch.cat((W, b.unsqueeze(dim=1)), dim=1)
        
        if self.data_dim   == 2:
            transformation = transformation.view(-1, 2, 3)
        elif self.data_dim == 3:
            transformation = transformation.view(-1, 3, 4)
        else:
            NotImplementedError('only support 2d and 3d')
        
        flow = F.affine_grid(transformation, moving.size(), align_corners=False)
        
        if self.data_dim   == 2:
            flow = flow.permute(0, 3, 1, 2)
        elif self.data_dim == 3:
            flow = flow.permute(0, 4, 1, 2, 3)
        else:
            NotImplementedError('only support 2d and 3d')
        
        affine_img = self.spatial_transformer(moving, flow)
        
        return A, affine_img 



class Elastic_Beta_VAE(nn.Module):
    def __init__(self,
                 input_ch  : int,
                 output_ch : int,
                 data_dim  : int,
                 latent_dim: int,
                 group_num : int,
                 img_shape : object = (256, 256),
                 filters   : object = [16, 32, 64, 128, 256]):
        super(Elastic_Beta_VAE, self).__init__()
        
        self.input_ch   = input_ch
        self.output_ch  = output_ch
        self.data_dim   = data_dim
        self.latent_dim = latent_dim
        self.group_num  = group_num
        self.image_shape= img_shape
        self.filters    = filters
        
        # Encoder Block
        self.encoder  = Encoder(self.input_ch, self.data_dim, self.latent_dim, self.group_num, self.filters)
        
        # Decoder Block
        self.decoder  = Decoder(self.input_ch, self.output_ch, self.data_dim, self.latent_dim, self.group_num, self.filters)

        # Spataial Transformer Network
        self.spatial_transformer = SpatialTransformer(self.image_shape)
        
        
    def forward(self, fixed: torch.tensor, moving: torch.tensor):
        # Concatenate fixed and moving images
        x = torch.cat((fixed, moving), dim=1)
        
        # Encoder Block
        distributions = self.encoder(x)
        
        # Decoder Block
        mu                = distributions[:, :self.latent_dim]
        logvar            = distributions[:, self.latent_dim:]
        z                 = reparametrize(mu, logvar)      
        deformation_field = self.decoder(z)
        deformation_field = deformation_field.view(moving.size())
        
        # Spatial Transformer
        elastic_img = self.spatial_transformer(moving, deformation_field)
               
        return deformation_field, elastic_img, mu, logvar
        


        
from torchsummary import summary

"""model = Beta_AE(data_dim = 2,
                z_latent_dim      = 256,
                 num_input_channels= 1,
                 num_outpt_channels= 1,
                 filters    = [32, 32, 32, 32, 32])
model = model.to('cuda')
print(model)
summary = summary(model,(1,  256, 256), device='cuda')"""

"""model = Affine_Beta_VAE(input_ch   = 2,
               data_dim   = 2,
               latent_dim = 256,
               img_shape  = (256, 256),
               filters    = [16, 32, 64, 128, 256])
model = model.to('cuda')
print(model)
summary = summary(model, [(1, 256, 256), (1, 256, 256)], device='cuda')"""

"""model = Elastic_Beta_VAE(input_ch   = 2,
                output_ch  = 1,
                data_dim   = 2,
                latent_dim = 256,
                img_shape  = (256, 256),
                filters    = [16, 32, 64, 128, 256])
model = model.to('cuda')
print(model)
summary = summary(model, [(1,  256, 256), (1,  256, 256)], device='cuda')"""

      