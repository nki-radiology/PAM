import torch
import torch.nn             as     nn
import torch.nn.functional  as     F
from   collections          import OrderedDict
from   networks.layer                import conv_layer
from   networks.layer                import conv_gl_avg_pool_layer
from   networks.network              import Encoder_WAE
from   networks.network              import Decoder
from   networks.spatial_transformer  import SpatialTransformer


class Wasserstein_AE(nn.Module):
    def __init__(self, 
                input_ch  : int = 1,
                input_dim : int = [256, 256, 512],
                latent_dim: int = 512,
                output_ch : int = 3,
                group_num : int = 8,
                filters   : object = [32, 64, 128, 256]
                 ):
        super(Wasserstein_AE, self).__init__()
        """
        Inputs:
            - num_input_channels: Number of input channels of the image. For medical images, this parameter usually is 1
            - z_latent_dim      : Dimensionality of the latent space (Z)
        """
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.output_ch  = output_ch
        self.group_num  = group_num
        self.filters    = filters
        
        # Encoder Block
        self.encoder  = Encoder_WAE(input_ch=self.input_ch, input_dim=self.input_dim, latent_dim=self.latent_dim,
                                    group_num=self.group_num, filters=self.filters)
        
        # Decoder Block
        self.decoder  = Decoder(output_ch=self.output_ch, input_dim=self.input_dim, latent_dim=self.latent_dim,
                                group_num=self.group_num, filters=self.filters)
        self.activation_func = nn.Tanh()
        
    
    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = self.activation_func(x_hat)
        return x_hat



class Affine_WAE(nn.Module):
    def __init__(self,
                 input_ch  : int,
                 data_dim  : int,
                 latent_dim: int,
                 img_shape : object = (256, 256),
                 filters   : object = [16, 32, 64, 128, 256]):
        super(Affine_WAE, self).__init__()
        """
        Inputs:
            - input_ch   : Number of input channels of the image. For medical images, this parameter usually is 1
            - latent_dim : Dimensionality of the latent space (Z)
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """          
        self.input_ch   = input_ch
        self.data_dim   = data_dim
        self.latent_dim = latent_dim
        self.image_shape= img_shape
        self.filters    = filters
        
        # Encoder Block
        self.encoder_net =  nn.Sequential(OrderedDict([
            ('affine_conv_16'    , conv_layer(self.data_dim)(in_channels=self.input_ch, out_channels=self.filters[0], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)), 
            ('affine_act_fn_16'  , nn.GELU()), # 256 x 256 => 128 x 128
            
            ('affine_conv_32'    , conv_layer(self.data_dim)(in_channels=self.filters[0], out_channels=self.filters[1], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('affine_act_fn_32'  , nn.GELU()), # 128 x 128 => 64 x 64
            
            ('affine_conv_64'    , conv_layer(self.data_dim)(in_channels=self.filters[1], out_channels=self.filters[2], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('affine_act_fn_64'  , nn.GELU()), # 64 x 64 => 32 x 32
            
            ('affine_conv_128'   , conv_layer(self.data_dim)(in_channels=self.filters[2], out_channels=self.filters[3], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('affine_act_fn_128' , nn.GELU()), # 32 x 32 => 16 x 16
            
            ('affine_conv_256'   , conv_layer(self.data_dim)(in_channels=self.filters[3], out_channels=self.filters[4], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
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



class Elastic_WAE(nn.Module):
    def __init__(self,
                 input_ch  : int,
                 output_ch : int,
                 data_dim  : int,
                 latent_dim: int,
                 img_shape : object = (256, 256),
                 filters   : object = [16, 32, 64, 128, 256]):
        super(Elastic_WAE, self).__init__()
        
        self.input_ch   = input_ch
        self.output_ch  = output_ch
        self.data_dim   = data_dim
        self.latent_dim = latent_dim
        self.image_shape= img_shape
        self.filters    = filters
        
        # Encoder Block
        self.encoder  = Encoder(self.input_ch, self.data_dim, self.latent_dim, self.filters)
        
        # Decoder Block
        self.decoder  = Decoder(self.input_ch, self.output_ch, self.data_dim, self.latent_dim, self.filters)

        # Spataial Transformer Network
        self.spatial_transformer = SpatialTransformer(self.image_shape)
    
    
    def forward(self, fixed: torch.tensor, moving: torch.tensor):
        # Concatenate fixed and moving images
        x = torch.cat((fixed, moving), dim=1)
        
        # Encoder Block
        z = self.encoder(x)
        
        # Decoder Block
        deformation_field = self.decoder(z)
        
        # Spatial Transformer
        elastic_img = self.spatial_transformer(moving, deformation_field)
               
        return deformation_field, elastic_img, z
        


#from torchsummary import summary
"""model = Affine_WAE(input_ch   = 2,
               data_dim   = 3,
               latent_dim = 256,
               img_shape  = (256, 256, 256),
               filters    = [16, 32, 64, 128, 256])
model = model.to('cuda')
print(model)
summary = summary(model, [(1, 256, 256, 256), (1, 256, 256, 256)], device='cuda')"""

"""model = Elastic_WAE(input_ch   = 2,
                output_ch  = 1,
                data_dim   = 2,
                latent_dim = 256,
                img_shape  = (256, 256),
                filters    = [16, 32, 64, 128, 256])
model = model.to('cuda')
print(model)
summary = summary(model, [(1,  256, 256), (1,  256, 256)], device='cuda')"""
