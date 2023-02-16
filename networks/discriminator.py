import torch
import torch.nn       as     nn
from   collections    import OrderedDict
from   networks.network import Encoder
from   networks.network import conv_gl_avg_pool_layer

class Discriminator(nn.Module):
    def __init__(self,
                 input_ch  : int = 1,
                 input_dim : int = [256, 256, 512],
                 latent_dim: int = 512,
                 group_num : int = 8,
                 filters   : object = [16, 32, 64, 128, 256]):
        super(Discriminator, self).__init__()
        """
        Inputs:
            - input_ch   : Number of input channels of the image. For medical images, this parameter usually is 1
            - latent_dim : Dimensionality of the latent space (Z)
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """          
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.group_num  = group_num
        self.filters    = filters
        features_linear_layer = 1024
        
        # Encoder Block
        self.conv_discriminator =  Encoder(input_ch=self.input_ch, input_dim=self.input_dim, latent_dim=self.latent_dim, group_num=self.group_num, filters=self.filters)
        
        self.linear_discriminator = nn.Sequential(OrderedDict([
            ('disc_gl_avg_pool' , conv_gl_avg_pool_layer(self.data_dim)(output_size=1)),
            ('disc_ft_vec_all'  , nn.Flatten()),
            ('disc_last_linear' , nn.Linear(in_features=self.filters[-1], out_features=features_linear_layer, bias=False)),
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
                         input_dim  = [256, 256, 512],
                         latent_dim = 512,
                         group_num  = 8,
                         filters    = [16, 32, 64, 128, 256])
model = model.to('cuda')
print(model)
summary = summary(model, [(1, 256, 256)], device='cuda')"""    