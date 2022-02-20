from  layers import *


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
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
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0, dim=self.dim)

        # activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

    def forward(self, x):
        y = self.conv1(x)       # convolution 1
        y = self.act1(y)        # activation 1
        if self.normalization:
            y = self.norm1(y)   # normalization 1
        y = self.conv2(y)       # convolution 2
        y = self.act2(y)        # activation 2
        if self.normalization:
            y = self.norm2(y)   # normalization 2

        before_pooling = y      # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)    # pooling
        return y, before_pooling


class UpBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 UpConvolution/Upsample.
    An activation follows each convolution.
    A normalization layer follows each convolution.
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

        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.normalization = normalization
        if conv_mode   == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
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
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # Activation layers
        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # Normalization layers
        if self.normalization:
            self.norm0 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

        # concatenate layer
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """ Forward pass
        Arguments:
            encoder_layer: Tensor from the encoder pathway
            decoder_layer: Tensor from the decoder pathway (to be up'd)
        """
        up_layer                         = self.up(decoder_layer)             # Up-convolution/up-sampling
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # Cropping

        if self.up_mode != 'transposed':
            # We need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer) # Convolution 0
        up_layer = self.act0(up_layer)      # Activation 0
        if self.normalization:
            up_layer = self.norm0(up_layer) # Normalization 0

        merged_layer = self.concat(up_layer, cropped_encoder_layer)  # concatenation
        y = self.conv1(merged_layer)        # Convolution 1
        y = self.act1(y)                    # Activation 1
        if self.normalization:
            y = self.norm1(y)               # Normalization 1
        y = self.conv2(y)                   # Convolution 2
        y = self.act2(y)                    # Activation 2
        if self.normalization:
            y = self.norm2(y)           # normalization 2
        return y


class UNet(nn.Module):
    """
    activation: 'relu', 'leaky', 'elu'
    normalization: 'batch', 'instance', 'group{group_size}'
    conv_mode: 'same', 'valid'
    dim: 2, 3
    up_mode: 'transposed', 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
    """
    def __init__(self,
                 in_channels    : int = 1,
                 out_channels   : int = 2,
                 n_blocks       : int = 4,
                 start_filters  : int = 32,
                 activation     : str = 'relu',
                 normalization  : str = 'batch',
                 conv_mode      : str = 'same',
                 dim            : int = 2,
                 up_mode        : str = 'transposed'
                 ):
        super().__init__()

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

        # create encoder path
        for i in range(self.n_blocks):
            num_filters_in  = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
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
        for i in range(n_blocks - 1):
            num_filters_in  = num_filters_out
            num_filters_out = num_filters_in // 2

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
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights= nn.init.xavier_uniform_,
                              method_bias   = nn.init.zeros_,
                              kwargs_weights= {},
                              kwargs_bias   = {}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)          # initialize bias

    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys()
                      if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'


model = UNet(in_channels  = 1,
             out_channels = 2,
             n_blocks     = 4,
             start_filters= 32,
             activation   = 'relu',
             normalization= 'group4',
             conv_mode    = 'same',
             dim          = 3)

x = torch.randn(size=(1, 1, 192, 192, 160), dtype=torch.float32)
with torch.no_grad():
    out = model(x)

print(f'Out: {out.shape}')

# from torchsummary import summary
# model.to(device)
# summary = summary(model, (1, 192, 192, 160))

