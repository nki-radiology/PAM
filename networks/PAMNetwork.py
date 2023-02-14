import torch
import torch.nn as nn


from networks.ViTDefNetwork             import ViTDefNetwork            # the choice of skip_connections is inside it
from networks.DeformationNetwork        import DeformationNetwork       # the choice of skip_connections is inside it   
from networks.AffineNetwork             import AffineNetwork
from torchsummary                       import summary


import sys
sys.path.append('../')

from general_config import affine, deformation, visiontransformer, general_choices

class PAMNetwork(nn.Module):

    def __init__(self):
        super(PAMNetwork, self).__init__()

        # Affine Network
        self.in_ch_aff   = affine.in_channels
        self.img_dim_aff = affine.img_dim
        self.affine_net = AffineNetwork(self.in_ch_aff, self.img_dim_aff)

        # Deformation Network     
        self.in_ch_def   = deformation.in_channels
        self.out_ch_def  = deformation.out_channels
        self.img_dim_def = deformation.img_dim
        self.filters     = deformation.filters
        self.skip_choice = deformation.skip_choice

        if general_choices.ViT_choice == 'no':
            self.deform_net  = DeformationNetwork(self.in_ch_def, self.out_ch_def, self.filters, self.img_dim_def)  

        if general_choices.ViT_choice == 'yes':   
            self.emb_size    = visiontransformer.emb_size
            self.num_heads   = visiontransformer.ViT_heads
            self.num_layers  = visiontransformer.ViT_layers    
            self.deform_net  = ViTDefNetwork(self.in_ch_def, self.out_ch_def, self.filters, self.img_dim_def, self.emb_size, self.num_heads, self.num_layers)  


    def forward(self, fixed: torch.tensor, moving: torch.tensor):
        """ forward function
        Compute the output tensors from the input tensors.

        :param fixed : fixed image
        :type fixed  : torch.tensor
        :param moving: moving image
        :type moving : torch.tensor

        :return      : the deformation field and the registered image
        :rtype       : torch.tensor
        """
        transform_0, warped_0 = self.affine_net(fixed, moving)
        transform_1, warped_1 = self.deform_net(fixed, warped_0)

        return transform_0, warped_0, transform_1, warped_1