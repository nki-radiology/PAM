import torch
import torch.nn       as     nn
from   collections    import OrderedDict
from   networks.layer import conv_layer
from   networks.layer import conv_gl_avg_pool_layer

class Discriminator(nn.Module):
    def __init__(self,
                 input_ch  : int,
                 data_dim  : int,
                 latent_dim: int,
                 group_num : int,
                 img_shape : object = (256, 256),
                 filters   : object = [16, 32, 64, 128, 256]):
        super(Discriminator, self).__init__()
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
        self.conv_discriminator =  nn.Sequential(OrderedDict([
            ('disc_conv_16'    , conv_layer(self.data_dim)(in_channels=self.input_ch, out_channels=self.filters[0], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)), 
            ('disc_gnorm_16'   , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[0])),
            ('disc_act_fn_16'  , nn.GELU()), # 256 x 256 => 128 x 128
            
            ('disc_conv_32'    , conv_layer(self.data_dim)(in_channels=self.filters[0], out_channels=self.filters[1], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('disc_gnorm_32'   , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[1])),
            ('disc_act_fn_32'  , nn.GELU()), # 128 x 128 => 64 x 64
            
            ('disc_conv_64'    , conv_layer(self.data_dim)(in_channels=self.filters[1], out_channels=self.filters[2], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('disc_gnorm_64'   , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[2])),
            ('disc_act_fn_64'  , nn.GELU()), # 64 x 64 => 32 x 32
            
            ('disc_conv_128'   , conv_layer(self.data_dim)(in_channels=self.filters[2], out_channels=self.filters[3], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('disc_gnorm_128'  , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[3])),
            ('disc_act_fn_128' , nn.GELU()), # 32 x 32 => 16 x 16
            
            ('disc_conv_256'   , conv_layer(self.data_dim)(in_channels=self.filters[3], out_channels=self.filters[4], 
                                                             kernel_size=3, stride=2, padding=1, bias=False)),
            ('disc_gnorm_256'  , nn.GroupNorm(num_groups=self.group_num, num_channels=self.filters[4])),
            ('disc_act_fn_256' , nn.GELU()), # 16 x 16 => 8 x 8
        ]))
        
        self.linear_discriminator = nn.Sequential(OrderedDict([
            ('disc_gl_avg_pool' , conv_gl_avg_pool_layer(self.data_dim)(output_size=1)),
            ('disc_ft_vec_all'  , nn.Flatten()),
            ('disc_last_linear' , nn.Linear(in_features=self.filters[4], out_features=1024, bias=False)),
            ('disc_last__act_fn', nn.Sigmoid()), 
        ]))
            
    
    def forward(self, x: torch.tensor):
                
        # Get last convolution
        fx = x = self.conv_discriminator(x)
        
        # Last layer
        x = self.linear_discriminator(x)
        
        return x, fx

from torchsummary import summary

"""model = Discriminator(input_ch   = 1,
               data_dim   = 2,
               latent_dim = 256,
               group_num  = 8,
               img_shape  = (256, 256),
               filters    = [16, 32, 64, 128, 256])
model = model.to('cuda')
print(model)
summary = summary(model, [(1, 256, 256)], device='cuda')"""    