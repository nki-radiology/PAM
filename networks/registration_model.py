import torch
import torch.nn            as     nn
from   networks.beta_vae_network    import Affine_Beta_VAE
from   networks.beta_vae_network    import Elastic_Beta_VAE 
from   networks.wasserstein_network import Affine_WAE
from   networks.wasserstein_network import Elastic_WAE

class Registration_Beta_VAE(nn.Module):   
    def __init__(self,
                 input_ch   : int,
                 output_ch  : int,
                 data_dim   : int,
                 latent_dim : int,
                 img_shape  : object = (256, 256),
                 filters    : object = [16, 32, 64, 128, 256]):
        
        super(Registration_Beta_VAE, self).__init__()
        
        self.input_ch   = input_ch
        self.output_ch  = output_ch
        self.data_dim   = data_dim
        self.latent_dim = latent_dim
        self.img_shape  = img_shape
        self.filters    = filters
        
        # Affine Network
        self.affine_net = Affine_Beta_VAE(input_ch=self.input_ch,
                                          data_dim=self.data_dim,
                                          latent_dim=self.latent_dim,
                                          img_shape=self.img_shape,
                                          filters=self.filters)
        
        # Deformation/Elastic Network
        self.elastic_net = Elastic_Beta_VAE(input_ch=self.input_ch,
                                            output_ch=self.output_ch,
                                            data_dim=self.data_dim,
                                            latent_dim=self.latent_dim,
                                            img_shape=self.img_shape,
                                            filters=self.filters)
        
    def forward(self, fixed:torch.tensor, moving:torch.tensor):
        transform_affine,  warped_affine               = self.affine_net(fixed, moving)
        transform_elastic, warped_elastic, mu, log_var = self.elastic_net(fixed, warped_affine)
        
        return transform_affine, warped_affine, transform_elastic, warped_elastic, mu, log_var
        


class Registration_Wasserstein_AE(nn.Module):   
    def __init__(self,
                 input_ch   : int,
                 output_ch  : int,
                 data_dim   : int,
                 latent_dim : int,
                 img_shape  : object = (256, 256),
                 filters    : object = [16, 32, 64, 128, 256]):
        
        super(Registration_Wasserstein_AE, self).__init__()
        
        self.input_ch   = input_ch
        self.output_ch  = output_ch
        self.data_dim   = data_dim
        self.latent_dim = latent_dim
        self.img_shape  = img_shape
        self.filters    = filters
        
        # Affine Network
        self.affine_net = Affine_WAE(input_ch=self.input_ch,
                                     data_dim=self.data_dim,
                                     latent_dim=self.latent_dim,
                                     img_shape=self.img_shape,
                                     filters=self.filters)
        
        # Deformation/Elastic Network
        self.elastic_net = Elastic_WAE(input_ch=self.input_ch,
                                       output_ch=self.output_ch,
                                       data_dim=self.data_dim,
                                       latent_dim=self.latent_dim,
                                       img_shape=self.img_shape,
                                       filters=self.filters)
        
    def forward(self, fixed:torch.tensor, moving:torch.tensor):
        transform_affine,  warped_affine     = self.affine_net(fixed, moving)
        transform_elastic, warped_elastic, z = self.elastic_net(fixed, warped_affine)
        
        return transform_affine, warped_affine, transform_elastic, warped_elastic, z
        
         