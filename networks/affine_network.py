import torch
import torch.nn            as nn
import torch.nn.functional as F
from collections      import OrderedDict
from networks.network import Encoder
from networks.network import conv_gl_avg_pool_layer
from networks.spatial_transfomer import SpatialTransformer

class Affine_Network(nn.Module):
    def __init__(self,
                    input_ch  : int = 2,
                    input_dim : int = [192, 192, 160],
                    group_num : int = 8,
                    filters   : object = [32, 64, 128, 256, 512]):

        super(Affine_Network, self).__init__()
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
        features_linear_layer = 1024
        
        # Encoder Block
        self.encoder_net =  Encoder(input_ch=self.input_ch, input_dim=self.input_dim, group_num=self.group_num, filters=self.filters)
                
        self.last_layer = nn.Sequential(OrderedDict([
            ('affine_gl_avg_pool' , conv_gl_avg_pool_layer(len(self.input_dim))(output_size=1)),
            ('affine_ft_vec_all'  , nn.Flatten()),
            ('affine_last_linear' , nn.Linear(in_features=self.filters[-1], out_features=features_linear_layer, bias=False)),
            ('affine_last_act_fn', nn.ReLU()), 
        ]))
        
        # Affine Transformation Blocks
        self.dense_w = nn.Sequential(OrderedDict([
            ('affine_w_matrix'       , nn.Linear(in_features=features_linear_layer, out_features=len(self.input_dim)**2, bias=False)), 
        ]))
        
        self.dense_b = nn.Sequential(OrderedDict([
            ('affine_b_vector'       , nn.Linear(in_features=features_linear_layer, out_features=len(self.input_dim), bias=False)), 
        ]))
        
        # Spatial Transformer
        self.spatial_transformer = SpatialTransformer(self.input_dim)
    
    
    def forward(self, fixed: torch.tensor, moving: torch.tensor):
        # Concatenate fixed and moving images
        x = torch.cat((fixed, moving), dim=1)
        
        # Encoding Block
        x = self.encoder_net(x)
        x = self.last_layer(x)
        
        # Get the degrees of freedom of the affine transformation
        W = self.dense_w(x).view(-1, len(self.input_dim), len(self.input_dim))
        b = self.dense_b(x).view(-1, len(self.input_dim))      
        
        # Input for the Spatial Transformer Network
        transformation = torch.cat((W, b.unsqueeze(dim=1)), dim=1)
        
        if len(self.input_dim)   == 2:
            transformation = transformation.view(-1, 2, 3)
        elif len(self.input_dim) == 3:
            transformation = transformation.view(-1, 3, 4)
        else:
            NotImplementedError('only support 2d and 3d')
        
        transformation = F.affine_grid(transformation, moving.size(), align_corners=False)
        
        if len(self.input_dim)   == 2:
            flow = transformation.permute(0, 3, 1, 2)
        elif len(self.input_dim) == 3:
            flow = transformation.permute(0, 4, 1, 2, 3)
        else:
            NotImplementedError('only support 2d and 3d')
        
        affine_registered_image = self.spatial_transformer(moving, flow)
        return transformation, affine_registered_image
    

'''from torchsummary import summary
model =  Affine_Network(input_ch = 2,
                    input_dim = [192, 192, 160],
                    group_num = 8,
                    filters   = [32, 64, 128, 256, 512])
summary = summary(model.to('cuda'), [(1, 192, 192, 160), (1, 192, 192, 160)])'''
