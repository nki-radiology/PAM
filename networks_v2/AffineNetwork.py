import torch
import torch.nn as nn
import torch.nn.functional as F
from   networks_v2.layers           import *
from   networks_v2.SpatialTransform import SpatialTransform


class EncodingBlock(nn.Module):
    """
    Encoding Block Module to perform the Affine transformation.
    It consists of a convolution, an activation after each convolution, and a normalization after each activation
    function.
    """

    def __init__(self,
                 in_channels    : int,
                 out_channels   : int,
                 pooling        : bool = False,
                 activation     : str  = 'relu',
                 normalization  : str  = None,
                 dim            : int  = 2,
                 conv_mode      : str  = 'same'):
        super().__init__()
        """ Encoding Block Initialization
        Returns the encoding block according to the assigned parameters.
        In case there is no specification of values in the parameters, the default values are considered.
    
        :param in_channels  : Number of channels-features in the input image
        :type  in_channels  : int
        :param out_channels : Number of channels-features produced by the convolution
        :type  out_channels : int
        :param pooling      : If true, it applies max pooling operation. Default: False
        :type  pooling      : bool
        :param activation   : Activation function after each convolution. Default: relu
        :type  activation   : str
        :param normalization: Normalization function after each activation. Default: None
        :type  normalization: str
        :param dim          : Dimension of the convolution. Default: 2
        :type  dim          : int
        :param conv_mode    : Convolution mode (same: padding=1, valid: padding=0). Default: same
        :type  conv_mode    : same
    
        :return: PAM encoding block 
        :rtype : nn.Module
        """
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.pooling       = pooling
        self.normalization = normalization

        if conv_mode       == 'same':
            self.padding   = 1
        elif conv_mode     == 'valid':
            self.padding   = 0
        self.dim           = dim
        self.activation    = activation

        # conv layers
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=self.padding,
                                    bias=True, dim=self.dim)

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0, dim=self.dim)

        # activation layers
        self.act1    = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

    def forward(self, x):
        """
        Forward function to compute the output tensor from the input tensor.
        :param x: Input tensor
        :type x: torch.tensor
        :return: Output tensor
        :rtype: torch.tensor
        """
        y = self.conv1(x)       # convolution   1
        y = self.act1(y)        # activation    1

        if  self.normalization:
            y = self.norm1(y)   # normalization 1

        if self.pooling:
            y = self.pool(y)    # pooling

        return y


class AffineNetwork(nn.Module):
    """ Affine Network
    It returns the affine network of PAM.
    """
    def __init__(self,
                 in_channels    : int = 1,
                 out_channels   : int = 2,
                 n_blocks       : int = 4,
                 start_filters  : int = 32,
                 activation     : str = 'relu',
                 normalization  : str = None,
                 conv_mode      : str = 'same',
                 dim            : int = 2,
                 ):
        super().__init__()

        """ Affine Network Initialization
        Returns the affine network module according to the assigned parameters.
        In case there is no specification of values in the parameters, the default values are considered.
    
        :param in_channels  : Number of channels-features in the input image
        :type  in_channels  : int
        :param out_channels : Number of channels-features produced by the convolution
        :type  out_channels : int
        :param n_blocks     : Encoding Blocks for the affine network. Default: 4
        :type  n_blocks     : int
        :param start_filters: Start filters number for convolutions. Default: 32
        :type  start_filters: bool
        :param activation   : Activation function after each convolution. Default: relu
        :type  activation   : str
        :param normalization: Normalization function after each activation. Default: None
        :type  normalization: str
        :param conv_mode    : Convolution mode (same: padding=1, valid: padding=0). Default: same
        :type  conv_mode    : same
        :param dim          : Dimension of the convolution. Default: 2
        :type  dim          : int
    
        :return: PAM Affine Network Module
        :rtype : nn.Module
        """

        self.in_channels     = in_channels
        self.out_channels    = out_channels
        self.n_blocks        = n_blocks
        self.start_filters   = start_filters
        self.activation      = activation
        self.normalization   = normalization
        self.conv_mode       = conv_mode
        self.dim             = dim
        self.encoding_blocks = []
        self.spatial_transf  = SpatialTransform((192, 192, 160))

        # Create the affine network according the number of encoding blocks
        for i in range(self.n_blocks):
            num_filters_in   = self.in_channels if i == 0 else num_filters_out
            num_filters_out  = self.start_filters * (2 ** i)
            pooling          = False

            encoding_block = EncodingBlock(in_channels  = num_filters_in,
                                           out_channels = num_filters_out,
                                           pooling      = pooling,
                                           activation   = self.activation,
                                           normalization= self.normalization,
                                           conv_mode    = self.conv_mode,
                                           dim          = self.dim)

            self.encoding_blocks.append(encoding_block)

        # End Encoding Block: global average pooling layer and fully connected layer
        self.global_avg_pool = get_global_avg_pool_layer(output_size=(3, 3, 2), dim=self.dim)
        self.end_dense_layer = nn.Sequential(
            nn.Linear(256*3*3*2, 1024),
            nn.ReLU()
        )

        # Affine Layers
        self.dense_w = nn.Sequential(
            nn.Linear(1024, 9)
        )

        self.dense_b = nn.Sequential(
            nn.Linear(1024, 3)
        )

        # Add the list of modules to current module
        self.encoding_blocks = nn.ModuleList(self.encoding_blocks)

        # Parameters initialization
        initialize_parameters(self.modules())


    def forward(self, fixed: torch.tensor, moving: torch.tensor):
        """ forward function
        Compute the output tensors from the input tensors.

        :param fixed : fixed image
        :type fixed  : torch.tensor
        :param moving: moving image
        :type moving : torch.tensor

        :return      : the affine matrix and the registered image
        :rtype       : torch.tensor
        """
        # Concatenate fixed and moving images
        x = torch.cat((fixed, moving), dim=1)

        # Encoding Blocks
        for module in self.encoding_blocks:
            x = module(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.end_dense_layer(x)


        # Affine Transformation: y = Wx + b
        if self.dim == 3:
            W = self.dense_w(x).view(-1, 3, 3)
            b = self.dense_b(x).view(-1, 3)
        else:
            W = self.dense(x).view(-1, 2, 2)
            b = self.dense_b(x).view(-1, 2)

        # Creating the Affine Matrix
        I = torch.eye(3, dtype=torch.float32, requires_grad=True , device="cuda")
        A = W + I

        # Input for the Spatial Transform Network
        transformation = torch.cat((W, b.unsqueeze(dim=1)), dim=1)
        transformation = transformation.view(-1, 3, 4)
        transformation = F.affine_grid(transformation, moving.size(), align_corners=False)

        if self.dim == 3:
            transformed = transformation.permute(0, 4, 1, 2, 3)
        else:
            transformed = transformation.permute(0, 3, 1, 2)

        # Spatial transform
        registered = self.spatial_transf(moving, transformed)

        return A, registered


"""
model = AffineNetwork(in_channels  = 2,
             out_channels = 1,
             n_blocks     = 6,
             start_filters= 8,
             activation   = 'relu',
             normalization= 'group8',
             conv_mode    = 'same',
             dim          = 3)

#x = torch.randn(size=(1, 1, 192, 192, 160), dtype=torch.float32)
#y = torch.randn(size=(1, 1, 192, 192, 160), dtype=torch.float32)
#with torch.no_grad():
#    out, flow = model(x, y)
    # out = model(x)

#print(f'Out: {out.shape}, Flow: {flow.shape}')

from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary = summary(model, [(1, 192, 192, 160), (1, 192, 192, 160)], device='cuda')
#summary = summary(model, (1, 192, 192, 160), device='cuda')
"""