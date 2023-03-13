import torch
import torch.nn         as     nn
from   collections      import OrderedDict
from   networks.network import Encoder
from   networks.network import conv_gl_avg_pool_layer


class Discriminator(nn.Module):
    def __init__(self,
                 input_ch  : int = 1,
                 input_dim : int = [192, 192, 160],
                 group_num : int = 8,
                 filters   : object = [8, 16, 32, 64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        """
        Inputs:
            - input_ch   : Number of input channels of the image. For medical images, this parameter usually is 1
            - latent_dim : Dimensionality of the latent space (Z)
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """          
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.group_num  = group_num
        self.filters    = filters
        out_features_last_layer = 1
        
        # Encoder Block
        self.conv_discriminator =  Encoder(input_ch=self.input_ch, input_dim=self.input_dim, group_num=self.group_num, filters=self.filters)
        
        self.linear_discriminator = nn.Sequential(OrderedDict([
            ('disc_gl_avg_pool' , conv_gl_avg_pool_layer(len(self.input_dim))(output_size=1)),
            ('disc_ft_vec_all'  , nn.Flatten()),
            ('disc_last_linear' , nn.Linear(in_features=self.filters[-1], out_features=out_features_last_layer, bias=False)),
            ('disc_last__act_fn', nn.Sigmoid()), 
        ]))
            
    
    def forward(self, x: torch.tensor):
        # Get last convolution
        fx = x = self.conv_discriminator(x)
        
        # Last layer
        x = self.linear_discriminator(x)
        
        return x, fx


'''from torchsummary import summary

model = Discriminator(input_ch   = 1,
                         input_dim  = [192, 192, 160],
                         group_num  = 8,
                         filters    = [8, 16, 32, 64, 128, 256, 512])
model = model.to('cuda')
print(model)
summary = summary(model, [(1, 192, 192, 160)], device='cuda') '''