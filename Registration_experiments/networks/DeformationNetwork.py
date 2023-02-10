import torch
import torch.nn as nn
from   networks.SpatialTransformer import SpatialTransformer
from   general_config              import deformation
from   collections                 import OrderedDict
import os

if deformation.skip_choice == 'yes':
    skip_choice = 1 #--Unet
elif deformation.skip_choice == 'no':
    skip_choice = 0 #--Autoencoder

"""
Convolution Class for the U-Net generator
"""

GN_number = 4
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

"""
Up sample Convolution Class
"""
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

"""
U-Net Generator class
"""
class DeformationNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, filters, img_dim):
        super(DeformationNetwork, self).__init__()
        self.in_ch   = in_channels   # 2
        self.out_ch  = out_channels  # 3
        
        # SMALL  NETWORK  FILTERS    : [4, 8, 16, 32, 64]
        # BIG    NETWORLK   FILTERS  : [16, 32, 64, 128, 256]
        
        self.filters = filters
        self.img_dim = img_dim       # (192,192,300)
        
        self.enc_1 = nn.Sequential(OrderedDict([            
            ('encoder_conv_1',         Conv(self.in_ch, self.filters[0]))
        ]))
        
        self.enc_2 = nn.Sequential(OrderedDict([            
            ('encoder_maxpool_1',      nn.MaxPool3d(kernel_size=2, stride=2, padding=0)),            
            ('encoder_conv_2',         Conv(self.filters[0], self.filters[1]))
        ]))
        
        self.enc_3 = nn.Sequential(OrderedDict([           
            ('encoder_maxpool_2',      nn.MaxPool3d(kernel_size=2, stride=2, padding=0)),            
            ('encoder_conv_3',         Conv(self.filters[1], self.filters[2]))
        ]))
        
        self.enc_4 = nn.Sequential(OrderedDict([            
            ('encoder_maxpool_3',      nn.MaxPool3d(kernel_size=2, stride=2, padding=0)),            
            ('encoder_conv_4',         Conv(self.filters[2], self.filters[3]))
        ]))
        
        self.enc_5 = nn.Sequential(OrderedDict([            
            ('encoder_maxpool_4',      nn.MaxPool3d(kernel_size=2, stride=2, padding=0)),            
            ('encoder_conv_5',         Conv(self.filters[3], self.filters[4]))
        ]))
        
        self.dec_upconv_5 = nn.Sequential(OrderedDict([            
            ('decoder_upconv_5',       Up_Conv(self.filters[4], self.filters[3]))
        ]))
        
        if skip_choice == 1:
            self.dec_conv_5 = nn.Sequential(OrderedDict([
            ('decoder_conv_5',         Conv(self.filters[4], self.filters[3]))
            ]))
        elif skip_choice == 0:
            self.dec_conv_5 = nn.Sequential(OrderedDict([
            ('decoder_conv_5',         Conv_noskip(self.filters[3]))
            ]))
        
        self.dec_upconv_4 = nn.Sequential(OrderedDict([            
            ('decoder_upconv_4',       Up_Conv(self.filters[3], self.filters[2]))
        ]))
        
        if skip_choice == 1:
            self.dec_conv_4 = nn.Sequential(OrderedDict([            
            ('decoder_conv_4',         Conv(self.filters[3], self.filters[2]))
            ]))
        elif skip_choice == 0:
            self.dec_conv_4 = nn.Sequential(OrderedDict([            
            ('decoder_conv_4',         Conv_noskip(self.filters[2]))
            ]))
        
        self.dec_upconv_3 = nn.Sequential(OrderedDict([            
            ('decoder_upconv_3',       Up_Conv(self.filters[2], self.filters[1]))
        ]))
        
        if skip_choice == 1:
            self.dec_conv_3 = nn.Sequential(OrderedDict([            
            ('decoder_conv_3',         Conv(self.filters[2], self.filters[1]))
            ]))
        elif skip_choice == 0:
            self.dec_conv_3 = nn.Sequential(OrderedDict([            
            ('decoder_conv_3',         Conv_noskip(self.filters[1]))
            ]))
        
        self.dec_upconv_2 = nn.Sequential(OrderedDict([            
            ('decoder_upconv_2',       Up_Conv(self.filters[1], self.filters[0]))
        ]))
        
        if skip_choice == 1:
            self.dec_conv_2 = nn.Sequential(OrderedDict([
            ('decoder_conv_2',         Conv(self.filters[1], self.filters[0]))
            ]))
        elif skip_choice == 0:
            self.dec_conv_2 = nn.Sequential(OrderedDict([
            ('decoder_conv_2',         Conv_noskip(self.filters[0]))
            ]))
        
        self.trans = nn.Sequential(OrderedDict([            
            ('transformation',         nn.Conv3d(self.filters[0], self.out_ch, kernel_size=1, stride=1, padding=0, bias=False))
        ]))
        
        self.spat_trs = SpatialTransformer(self.img_dim)       


    def forward(self, fixed, moving):
        
        x = torch.cat((fixed, moving), dim=1) 
        e1 = self.enc_1(x)
        e2 = self.enc_2(e1)
        e3 = self.enc_3(e2)
        e4 = self.enc_4(e3)
        e5 = self.enc_5(e4)
        
        d5 = self.dec_upconv_5(e5) 
        d5 = autoexpand(e4, d5)
        if skip_choice == 1:
            d5  = torch.cat((e4, d5), dim=1)        
        d5 = self.dec_conv_5(d5)
        
        d4 = self.dec_upconv_4(d5)
        d4 = autoexpand(e3, d4)
        if skip_choice == 1:
            d4  = torch.cat((e3, d4), dim=1)
        d4 = self.dec_conv_4(d4)
        
        d3 = self.dec_upconv_3(d4)
        d3 = autoexpand(e2, d3)
        if skip_choice == 1:
            d3  = torch.cat((e2, d3), dim=1)
        d3 = self.dec_conv_3(d3)
        
        d2 = self.dec_upconv_2(d3)
        d2 = autoexpand(e1, d2)
        if skip_choice == 1:
            d2  = torch.cat((e1, d2), dim=1)
        d2 = self.dec_conv_2(d2)
        
        deformation_field = self.trans(d2) 
        registered_img = self.spat_trs(moving, deformation_field)
               
        return deformation_field, registered_img

    
# To summarize the complete model
"""
from torchsummary import summary
model = DeformationNetwork(2,1,[16, 32, 64, 128, 256],(192,192,300))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

x = torch.randn(size=(1, 1, 192, 192, 300), dtype=torch.float32, device=device)
y = torch.randn(size=(1, 1, 192, 192, 300), dtype=torch.float32, device=device)
with torch.no_grad():
    transformation0, registered0 = model(x, y)

print(f'Out: {transformation0.shape}, Flow: {registered0.shape}')
"""

"""
# To summarize the complete model
from torchsummary import summary
model  = DeformationNetwork(2,1,[16, 32, 64, 128, 256],(192,192,300))
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
summary = summary(model, [(1, 192, 192, 300), (1, 192, 192, 300)], device='cuda')
"""