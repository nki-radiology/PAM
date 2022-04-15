import torch
import torch.nn as nn
from networks_v2.DeformationNetwork import DeformationNetwork
from networks_v2.AffineNetwork import AffineNetwork
from torchsummary import summary

class RegistrationNetwork(nn.Module):

    def __init__(self,
                 in_channels    : int = 1,
                 out_channels   : int = 2,
                 n_blocks_affine: int = 5,
                 n_blocks_deform: int = 5,
                 start_filters_a: int = 8,
                 start_filters_d: int = 16,
                 activation     : str = 'relu',
                 normalization  : str = 'batch',
                 conv_mode      : str = 'same',
                 dim            : int = 2,
                 ):
        super().__init__()
        """ PAM MODEL
        This model represents the PAM Model
    
        :param in_channels    : Number of channels-features in the input image.    Default: 1
        :type  in_channels    : int
        :param out_channels   : Number of channels-features produced by the model. Default: 1
        :type  out_channels   : int
        :param n_blocks_affine: Block Number for the affine network. Default: 5
        :type  n_blocks_affine: int
        :param n_blocks_deform: Block Number for the deformation network. Default: 5
        :type  n_blocks_deform: int
        :param start_filters  : Start filters number for convolutions. Default: 16
        :type  start_filters  : bool
        :param activation     : Activation function after each convolution. Default: relu
        :type  activation     : str
        :param normalization  : Normalization function after each activation. Default: None
        :type  normalization  : str
        :param conv_mode      : Convolution mode (same: padding=1, valid: padding=0). Default: same
        :type  conv_mode      : same
        :param dim            : Dimension of the convolution. Default: 2
        :type  dim            : int
    
        :return: PAM Model
        :rtype : nn.Module
        """
        self.in_channels     = in_channels
        self.out_channels    = out_channels
        self.n_blocks_affine = n_blocks_affine
        self.n_blocks_deform = n_blocks_deform
        self.start_filters_a = start_filters_a
        self.start_filters_d = start_filters_d
        self.activation      = activation
        self.normalization   = normalization
        self.conv_mode       = conv_mode
        self.dim             = dim

        self.affine_net = AffineNetwork(in_channels        = self.in_channels,
                                        out_channels       = self.out_channels,
                                        n_blocks           = self.n_blocks_affine,
                                        start_filters      = self.start_filters_a,
                                        activation         = self.activation,
                                        normalization      = self.normalization,
                                        conv_mode          = self.conv_mode,
                                        dim                = self.dim)

        self.deform_net = DeformationNetwork(in_channels   = self.in_channels,
                                             out_channels  = self.out_channels,
                                             n_blocks      = self.n_blocks_deform,
                                             start_filters = self.start_filters_d,
                                             activation    = self.activation,
                                             normalization = self.normalization,
                                             conv_mode     = self.conv_mode,
                                             dim           = self.dim)

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
        transform_0, warped_0, = self.affine_net(fixed, moving)
        transform_1, warped_1 = self.deform_net(fixed, warped_0)

        return transform_0, warped_0, transform_1, warped_1


#

"""
model = RegistrationNetwork(in_channels  = 2,
             out_channels = 1,
             n_blocks_affine= 4,
             n_blocks_deform= 4,
             start_filters= 16,
             activation   = 'relu',
             normalization= 'group4',
             conv_mode    = 'same',
             dim          = 3)


x = torch.randn(size=(1, 1, 192, 192, 160), dtype=torch.float32)
y = torch.randn(size=(1, 1, 192, 192, 160), dtype=torch.float32)
with torch.no_grad():
    x_0, y_0, x_1, y_1 = model(x, y)
    # out = model(x)

print(f'Out: {x_0.shape}, Flow: {y_0.shape}')

from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary = summary(model, [(1, 192, 192, 160), (1, 192, 192, 160)], device='cuda')
#summary = summary(model, (1, 192, 192, 160), device='cuda')
"""
