import torch
import torch.nn as nn

GN_number = 4

""" Convolution Class for the U-Net generator """
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d   (in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm( num_groups=GN_number,     num_channels=out_ch),
            nn.ReLU     (inplace=True),
            nn.Conv3d   (in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm( num_groups=GN_number,      num_channels=out_ch),
            nn.ReLU     (inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

""" Up sample Convolution Class """
class Up_Conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Up_Conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample (scale_factor=2,     mode        ='trilinear'),
            nn.Conv3d   (in_channels =in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups =GN_number,      num_channels=out_ch),
            nn.ReLU     (inplace=True)
        )

    def forward(self, x):
        out = self.up(x)
        return out

""" Convolution Class for decoder part, just a single convolution """
class Conv_noskip(nn.Module):
    def __init__(self, out_ch):
        super(Conv_noskip, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d   (in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm( num_groups=GN_number,      num_channels=out_ch),
            nn.ReLU     (inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
    
def autoexpand(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):

    """ autoexpand
    Helper function that allows to add a layer in the decoder layers if their sizes do not match with the
    corresponding layers from the encoder block.
    For example, if I have 1,32,24,24,37 in the encoder and 1,32,24,24,36 in the decoder, it adds a 1,32,24,24,1 layer to the decoder part."""

    device = torch.device('cuda:0')
    if encoder_layer.shape[4] != decoder_layer.shape[4]:
        dim_to_add = (encoder_layer.shape[4] - decoder_layer.shape[4])
        layer_to_add = torch.zeros([encoder_layer.shape[0],
                                    encoder_layer.shape[1],
                                    encoder_layer.shape[2],
                                    encoder_layer.shape[3],
                                    dim_to_add],device=device)
        decoder_layer = torch.cat((decoder_layer,layer_to_add),dim=4)
    return decoder_layer

def autoconcat(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor, filters_to_have_after_concat, filters_before_concat):     
    """
        input: filters_to_have_after_concat = filters[4] = 64 -- just as example in the first step of the decoder pathway
        input: filter_before_concat = decoder_layer.shape[1]  -- always true
    
    This function is needed when the filters defined for the network are not in this shape" [n, 2n, 4n, 8n, ...].
    It's needed for the skip connection, since after the concatenation there is another convolution (Up_conv), and the number of filters expected 
    in input in the convolution is not equal to the number of filters after the concatenation.
    So, this function does a "cropped concatenation", saving only a part of the encoder layer and a part of the decoder one (the central ones). 
    
    """    
    if filters_to_have_after_concat != filters_before_concat*2:
        center_enc = int(encoder_layer.shape[1]/2)
        center_dec = int(decoder_layer.shape[1]/2)
        encoder_layer = encoder_layer[:,center_enc - int(filters_to_have_after_concat/4) : center_enc + int(filters_to_have_after_concat/4) ,:,:,:]
        decoder_layer = decoder_layer[:,center_dec - int(filters_to_have_after_concat/4) : center_dec + int(filters_to_have_after_concat/4) ,:,:,:]
    
    skip_connected_layer = torch.cat((encoder_layer,decoder_layer), dim=1)
    return skip_connected_layer