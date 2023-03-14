import torch
import torch.nn as nn
from collections                  import OrderedDict
from networks.network             import conv_gl_avg_pool_layer
from networks.affine_network      import Affine_Network
from networks.deformation_network import Deformation_Network 
class Registration_PAM(nn.Module):   
    def __init__(self,
                 input_ch   : int = 2,
                 input_dim  : int = [192, 192, 160],
                 latent_dim : int = 512,
                 output_ch  : int = 3,
                 group_num  : int = 8,
                 filters    : object = [32, 64, 128, 256, 512]):
        super(Registration_PAM, self).__init__()
        
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.output_ch  = output_ch
        self.group_num  = group_num
        self.filters    = filters
        
        # Affine Network
        self.affine_net = Affine_Network(input_ch   = self.input_ch,
                                         input_dim  = self.input_dim,
                                         group_num  = self.group_num,
                                         filters    = self.filters)

        # Deformation/Elastic Network
        self.deformation_net = Deformation_Network(input_ch   = self.input_ch,
                                                   input_dim  = self.input_dim,
                                                   latent_dim = self.latent_dim,
                                                   output_ch  = self.output_ch,
                                                   group_num  = self.group_num,
                                                   filters    = self.filters)
        
    def forward(self, fixed:torch.tensor, moving:torch.tensor):
        transform_affine,  warped_affine     = self.affine_net(fixed, moving)
        transform_elastic, warped_elastic, _ = self.deformation_net(fixed, warped_affine)
        return transform_affine, warped_affine, transform_elastic, warped_elastic
        


class Registration_PAM_Survival(nn.Module):   
    def __init__(self,
                 input_ch   : int = 2,
                 input_dim  : int = [256, 256, 512],
                 latent_dim : int = 512,
                 output_ch  : int = 3,
                 group_num  : int = 8,
                 filters    : object = [32, 64, 128, 256]):
        
        super(Registration_PAM_Survival, self).__init__()
        
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.output_ch  = output_ch
        self.group_num  = group_num
        self.filters    = filters
        out_features_last_layer = 1
        
        # Affine Network
        self.affine_net = Affine_Network(input_ch   = self.input_ch,
                                         input_dim  = self.input_dim,
                                         group_num  = self.group_num,
                                         filters    = self.filters)

        # Deformation/Elastic Network
        self.deformation_net = Deformation_Network(input_ch   = self.input_ch,
                                                   input_dim  = self.input_dim,
                                                   latent_dim = self.latent_dim,
                                                   output_ch  = self.output_ch,
                                                   group_num  = self.group_num,
                                                   filters    = self.filters)
        
        self.survival_layer = nn.Sequential(OrderedDict([
            ('disc_gl_avg_pool' , conv_gl_avg_pool_layer(len(self.input_dim))(output_size=1)),
            ('disc_ft_vec_all'  , nn.Flatten()),
            ('disc_last_linear' , nn.Linear(in_features=self.filters[-1], out_features=out_features_last_layer, bias=False)),
            ('disc_last__act_fn', nn.Sigmoid()), 
        ]))
        
    def forward(self, fixed:torch.tensor, moving:torch.tensor):
        transform_affine,  warped_affine     = self.affine_net(fixed, moving)
        transform_elastic, warped_elastic, Z = self.deformation_net(fixed, warped_affine)
        survival_output                      = self.survival_layer(Z)
        return transform_affine, warped_affine, transform_elastic, warped_elastic, survival_output
        
         