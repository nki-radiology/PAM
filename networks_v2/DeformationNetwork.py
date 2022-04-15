import torch
from   networks_v2.layers           import *
from   networks_v2.SpatialTransform import SpatialTransform

class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x


class DownBlock(nn.Module):
    """
    Down Block Module to perform the Elastic deformation.
    It consists of a convolution, an activation after each convolution, and a normalization after each activation
    function.
    """

    def __init__(self,
                 in_channels    : int,
                 out_channels   : int,
                 pooling        : bool = True,
                 activation     : str  = 'relu',
                 normalization  : str  = None,
                 dim            : int  = 2,         # str = 2
                 conv_mode      : str  = 'same'):
        super().__init__()
        """ Down Block Initialization
        Returns the encoding block of the UNet model according to the assigned parameters.
        In case there is no specification of values in the parameters, the default values are considered.
    
        :param in_channels  : Number of channels-features in the input image
        :type  in_channels  : int
        :param out_channels : Number of channels-features produced by the convolution
        :type  out_channels : int
        :param pooling      : If true, it applies max pooling operation. Default: True
        :type  pooling      : bool
        :param activation   : Activation function after each convolution. Default: relu
        :type  activation   : str
        :param normalization: Normalization function after each activation. Default: None
        :type  normalization: str
        :param dim          : Dimension of the convolution. Default: 2
        :type  dim          : int
        :param conv_mode    : Convolution mode (same: padding=1, valid: padding=0). Default: same
        :type  conv_mode    : str
    
        :return: UNet Down block 
        :rtype : nn.Module
        """

        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.pooling       = pooling
        self.normalization = normalization

        if conv_mode   == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim        = dim
        self.activation = activation

        # conv layers
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)
        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0, dim=self.dim)

        # activation layers
        self.act1 = get_activation(self.activation)

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
        y = self.conv1(x)       # convolution 1
        y = self.act1(y)        # activation 1
        if self.normalization:
            y = self.norm1(y)   # normalization 1

        before_pooling = y      # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)    # pooling
        return y, before_pooling


class UpBlock(nn.Module):
    """
    Up Block Module to perform the Elastic deformation.
    It consists of a convolution, an activation after each convolution, and a normalization after each activation
    function.
    """

    def __init__(self,
                 in_channels    : int,
                 out_channels   : int,
                 activation     : str = 'relu',
                 normalization  : str = None,
                 dim            : int = 3,
                 conv_mode      : str = 'same',
                 up_mode        : str = 'transposed'
                 ):
        super().__init__()
        """ Up Block Initialization
        Returns the decoding block of the UNet model according to the assigned parameters.
        In case there is no specification of values in the parameters, the default values are considered.
    
        :param in_channels  : Number of channels-features in the input image
        :type  in_channels  : int
        :param out_channels : Number of channels-features produced by the convolution
        :type  out_channels : int
        :param activation   : Activation function after each convolution. Default: relu
        :type  activation   : str
        :param normalization: Normalization function after each activation. Default: None
        :type  normalization: str
        :param dim          : Dimension of the convolution. Default: 2
        :type  dim          : int
        :param conv_mode    : Convolution mode (same: padding=1, valid: padding=0). Default: same
        :type  conv_mode    : str
        :param up_mode      : Convolution mode. Default: transposed
        :type  up_mode      : str
        
        :return: UNet Up block 
        :rtype : nn.Module
        """
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.normalization = normalization

        if conv_mode   == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0

        self.dim        = dim
        self.activation = activation
        self.up_mode    = up_mode

        # Up-convolution/ Up-sampling layer
        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, dim=self.dim,
                               up_mode=self.up_mode)

        # conv layers
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                                    bias=True, dim=self.dim)
        self.conv1 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                    padding=self.padding,
                                    bias=True, dim=self.dim)

        # Activation layers
        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)

        # Normalization layers
        if self.normalization:
            self.norm0 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
        # concatenate layer
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """
        Forward function to compute the output tensor from the input tensor.
        :param encoder_layer: Tensor from the encoder pathway
        :type encoder_layer: torch.tensor
        :param decoder_layer: Tensor from the decoder pathway
        :type decoder_layer: torch.tensor

        :return: Output tensor
        :rtype: torch.tensor
        """
        up_layer                         = self.up(decoder_layer)             # Up-convolution/up-sampling
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # Cropping

        if self.up_mode != 'transposed':
            # We need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer)                         # Convolution 0
        up_layer = self.act0(up_layer)                              # Activation  0
        if self.normalization:
            up_layer = self.norm0(up_layer)  # Normalization 0
        merged_layer = self.concat(up_layer, cropped_encoder_layer)  # concatenation
        y = self.conv1(merged_layer)                                 # Convolution   1
        y = self.act1(y)                                             # Activation    1
        if self.normalization:
            y = self.norm1(y)                                        # Normalization 1
        return y


class DeformationNetwork(nn.Module):
    """ Elastic Deformation Network
    It returns the Elastic Deformation of PAM.
    This network follows a UNet architecture.
    """
    def __init__(self,
                 in_channels    : int = 1,
                 out_channels   : int = 1,
                 n_blocks       : int = 4,
                 start_filters  : int = 32,
                 activation     : str = 'relu',
                 normalization  : str = 'batch',
                 conv_mode      : str = 'same',
                 dim            : int = 3,
                 up_mode        : str = 'transposed'
                 ):
        super().__init__()
        """ Elastic Deformation Network Initialization
        Returns the elastic deformation network module according to the assigned parameters.
        In case there is no specification of values in the parameters, the default values are considered.
        The activation function could be 'relu', 'leaky' or 'elu'
        The type of normalization could be 'batch', 'instance' or 'group{group_size}'
        The convolution mode could be 'same' or 'valid'
        The dim could be 2, 3
        The up_mode could be 'transposed', 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
    
        :param in_channels  : Number of channels-features in the input image.    Default: 1
        :type  in_channels  : int
        :param out_channels : Number of channels-features produced by the model. Default: 1
        :type  out_channels : int
        :param n_blocks     : Block Number for the deformation network. Default: 4
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
    
        :return: PAM Elastic Deformation Network Module
        :rtype : nn.Module
        """
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.n_blocks       = n_blocks
        self.start_filters  = start_filters
        self.activation     = activation
        self.normalization  = normalization
        self.conv_mode      = conv_mode
        self.dim            = dim
        self.up_mode        = up_mode

        self.down_blocks    = []
        self.up_blocks      = []
        self.features       = [16, 32, 64, 80, 96, 128]

        self.spatial_transf = SpatialTransform((192, 192, 160))

        # create encoder path
        num_feature = len(self.features)
        for i in range(num_feature):
            num_filters_in  = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.features[i]
            pooling = True if i < self.n_blocks - 1 else False

            down_block = DownBlock(in_channels  = num_filters_in,
                                   out_channels = num_filters_out,
                                   pooling      = pooling,
                                   activation   = self.activation,
                                   normalization= self.normalization,
                                   conv_mode    = self.conv_mode,
                                   dim          = self.dim)

            self.down_blocks.append(down_block)

        # create decoder path (requires only n_blocks-1 blocks)
        start_idx = num_feature - 2
        for i in range(num_feature - 1):
            num_filters_in  = num_filters_out
            num_filters_out = self.features[start_idx - i]

            up_block = UpBlock(in_channels  = num_filters_in,
                               out_channels = num_filters_out,
                               activation   = self.activation,
                               normalization= self.normalization,
                               conv_mode    = self.conv_mode,
                               dim          = self.dim,
                               up_mode      = self.up_mode)

            self.up_blocks.append(up_block)

        # final convolution
        self.conv_final = get_conv_layer(num_filters_out, self.out_channels, kernel_size=1, stride=1, padding=0,
                                         bias=True, dim=self.dim)

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks   = nn.ModuleList(self.up_blocks)

        # initialize the weights
        initialize_parameters(self.modules())


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
        # Concatenate fixed and moving images
        x = torch.cat((fixed, moving), dim=1)

        encoder_output = []

        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        transformation = self.conv_final(x)

        # Spatial Transform
        registered_img = self.spatial_transf(moving, transformation)

        return transformation, registered_img


"""
model = DeformationNetwork(in_channels  = 2,
                           out_channels = 1,
                           n_blocks     = 6,
                           start_filters= 16,
                           activation   = 'relu',
                           normalization= 'group4',
                           conv_mode    = 'same',
                           dim          = 3)


x = torch.randn(size=(1, 1, 192, 192, 160), dtype=torch.float32)
y = torch.randn(size=(1, 1, 192, 192, 160), dtype=torch.float32)
with torch.no_grad():
    reg, tra = model(x, y)
    # out = model(x)

print(f'Reg: {reg.shape}, Tra: {tra.shape}')

from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary = summary(model, [(1, 192, 192, 160), (1, 192, 192, 160)], device='cuda')
#summary = summary(model, (1, 192, 192, 160), device='cuda')
"""