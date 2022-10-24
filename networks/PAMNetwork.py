import torch
import torch.nn as nn
from networks.DeformationNetwork import DeformationNetwork
from networks.AffineNetwork      import AffineNetwork
from torchsummary import summary

import sys
sys.path.append('../')
from config import affine
from config import deformation


class PAMNetwork(nn.Module):

    def __init__(self):
        super(PAMNetwork, self).__init__()

        # Affine Network
        self.in_ch_aff   = affine.in_channels
        self.filters_aff = affine.filters
        self.img_dim_aff = affine.img_dim
        self.affine_net = AffineNetwork(self.in_ch_aff, self.filters_aff, self.img_dim_aff)

        # Deformation Network
        self.in_ch_def   = deformation.in_channels
        self.out_ch_def  = deformation.out_channels
        self.filters_def = deformation.filters
        self.img_dim_def = deformation.img_dim
        self.deform_net  = DeformationNetwork(self.in_ch_def, self.out_ch_def, self.filters_def, self.img_dim_def)


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



"""
# To summarize the complete model
from torchsummary import summary
model = PAMNetwork()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

x = torch.randn(size=(1, 1, 192, 192, 160), dtype=torch.float32, device='cuda')
y = torch.randn(size=(1, 1, 192, 192, 160), dtype=torch.float32, device='cuda')
with torch.no_grad():
    x_0, y_0, x_1, y_1 = model(x, y)

print(f'Out: {x_0.shape}, Flow: {y_0.shape}')


#summary = summary(model, [(1, 192, 192, 160), (1, 192, 192, 160)])
"""
