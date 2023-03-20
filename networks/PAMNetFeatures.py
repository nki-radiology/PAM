import torch
import torch.nn as nn
from networks.DeformationNetFeatures import DeformationNetwork
from networks.AffineNetwork      import AffineNetwork
from torchsummary import summary

import sys
sys.path.append('../')
from config import affine
from config import deformation
from config import image

class PAMNetwork(nn.Module):

    def __init__(self):
        super(PAMNetwork, self).__init__()
        
        self.img_dim = image.img_dim

        # Affine Network
        self.filters_aff = affine.filters
        self.affine_net = AffineNetwork(self.filters_aff, self.img_dim)
        
        # Deformation Network
        self.filters_def = deformation.filters
        self.deform_net  = DeformationNetwork(self.filters_def, self.img_dim)

    def forward_init(self, fixed: torch.tensor, moving: torch.tensor):
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
        transform_0, warped_0             = self.affine_net(fixed, moving)
        e4, e5, d4, transform_1, warped_1 = self.deform_net(fixed, warped_0)

        return transform_0, warped_0, e4, e5, d4, transform_1, warped_1


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
