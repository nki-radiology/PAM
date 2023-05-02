import torch
import torch.nn as nn
from networks.network import Conv
from networks.network import Up_Conv
from networks.network import conv_layer
from networks.network import max_pool_layer
from networks.spatial_transfomer import SpatialTransformer

class Deformation_Network(nn.Module):
    def __init__(self,
                 input_ch  : int = 2,
                 input_dim : int = [192, 192, 160],
                 latent_dim: int = 512,
                 output_ch : int = 3,
                 group_num : int = 8,
                 filters   : object = [32, 64, 128, 256, 512]
                ):
        super(Deformation_Network, self).__init__()
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.output_ch  = output_ch
        self.group_num  = group_num
        self.filters    = filters
        
        # Encoder Block
        # Filters sequence
        seq = [self.input_ch] + self.filters
        self.convnet = nn.ModuleDict()
        # Convolutions layers
        for i, (j, k) in enumerate(zip(seq, seq[1:]), start=1):
            self.convnet['encoder_conv_' + str(i)] = Conv(input_ch=j, output_ch=k, group_num=self.group_num, dim=len(self.input_dim))
        # Max pooling layers. Considering from 2 because there is no max pooling after the input_ch nor after the last convolution
        for i, _ in enumerate(seq[2:], start=1):
            self.convnet['encoder_max_pool_' + str(i)] = max_pool_layer(len(self.input_dim))(kernel_size=2, stride=2)
        
        # Decoder Block
        # Filters sequence
        seq = list(reversed(self.filters))
        # Up convolution layers
        for i, (j, k) in enumerate(zip(seq, seq[1:]), start=1):
            self.convnet['decoder_up_conv_' + str(i)] = Up_Conv(input_ch=j, output_ch=k, group_num=self.group_num, dim=len(self.input_dim))
        # Convolution layers
        for i, (j, k) in enumerate(zip(seq, seq[1:]), start=1):
            self.convnet['decoder_conv_' + str(i)] = Conv(input_ch=j, output_ch=k, group_num=self.group_num, dim=len(self.input_dim))
        
        # Deformation field and registered image
        self.convnet['transformation'] = conv_layer(len(self.input_dim))(in_channels=self.filters[0], out_channels=self.output_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.spatial_transformer = SpatialTransformer(self.input_dim) 
    
    
    def forward(self, fixed: torch.tensor, moving: torch.tensor):
        
        x = torch.cat((fixed, moving), dim=1) 
        e1 = self.convnet['encoder_conv_1'](x)

        e2 = self.convnet['encoder_max_pool_1'](e1)
        e2 = self.convnet['encoder_conv_2'](e2)

        e3 = self.convnet['encoder_max_pool_2'](e2)
        e3 = self.convnet['encoder_conv_3'](e3)

        e4 = self.convnet['encoder_max_pool_3'](e3) 
        e4 = self.convnet['encoder_conv_4'](e4)

        # Latent space: useful for survival
        e5 = self.convnet['encoder_max_pool_4'](e4)
        e5 = self.convnet['encoder_conv_5'](e5) 
        
        d5 = self.convnet['decoder_up_conv_1'](e5)         

        d5 = torch.cat((e4, d5), dim=1)        
        d5 = self.convnet['decoder_conv_1'](d5)
        
        d4 = self.convnet['decoder_up_conv_2'](d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.convnet['decoder_conv_2'](d4)
        
        d3 = self.convnet['decoder_up_conv_3'](d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.convnet['decoder_conv_3'](d3)
        
        d2 = self.convnet['decoder_up_conv_4'](d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.convnet['decoder_conv_4'](d2)
        
        deformation_field = self.convnet['transformation'](d2) 
        registered_img    = self.spatial_transformer(moving, deformation_field)
               
        return deformation_field, registered_img, e5 


'''from torchsummary import summary
model =  Deformation_Network(input_ch = 2,
                    input_dim = [192, 192, 160],
                    latent_dim= 512,
                    output_ch = 3,
                    group_num = 8,
                    filters  = [16, 32, 64, 128, 256])
summary = summary(model.to('cuda'), [(1, 192, 192, 160), (1, 192, 192, 160)])'''