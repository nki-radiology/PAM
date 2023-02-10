import torch
import torch.nn as nn
import os
from   general_config              import deformation, visiontransformer
from   collections                 import OrderedDict
from   networks.SpatialTransformer import SpatialTransformer         
from   networks.ViTransformer      import *                     

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

"""
Convolution Class for decoder part, just a single convolution
""" 
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
class ViTDefNetwork(nn.Module):
    
    def __init__(self, in_channels, out_channels, filters, img_dim, emb_size, num_heads, num_layers):
        super(ViTDefNetwork, self).__init__()
        self.in_ch   = in_channels   # 2
        self.out_ch  = out_channels  # 3
        
        # SMALL  NETWORK  FILTERS    : [4, 8, 16, 32, 64]
        # BIG    NETWORLK   FILTERS  : [16, 32, 64, 128, 256]
        
        self.filters = filters
        self.img_dim = img_dim                                  # (192,192,300)
        self.emb_size = visiontransformer.emb_size              # 4096 if big_net, 1024 if small_net   
        self.num_heads = visiontransformer.ViT_heads            # default = 16
        self.num_layers = visiontransformer.ViT_layers          # default = 12
        
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
        
        self.ViT = Transformer(emb_size    = self.emb_size, 
                               in_channels = self.filters[4], 
                               patch_size  = 6,
                               num_heads   = self.num_heads,
                               num_layers  = self.num_layers
                              ) 
        self.conv_vit = nn.Sequential(
            nn.Upsample (scale_factor = 6, mode = 'trilinear'),
            nn.Conv3d   (in_channels  = self.emb_size, out_channels = self.filters[4], kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(num_groups   = GN_number, num_channels = self.filters[4]),
            nn.ReLU     (inplace      = True))
                           
    def forward(self, fixed, moving): 
        
        x = torch.cat((fixed, moving), dim=1) 
        e1 = self.enc_1(x)
        e2 = self.enc_2(e1)
        e3 = self.enc_3(e2)
        e4 = self.enc_4(e3)
        e5 = self.enc_5(e4)
        
        # Application of the Transformer to the feature-map of dimensions (1, 256, 12, 12, 18)
        vit_in  = torch.squeeze(e5)                                         # vit_in : (256, 12, 12, 18)
        h, w, l    = vit_in.size(1), vit_in.size(2), vit_in.size(3)
        vit_out = self.ViT(e5)                                              # vit_out : (B, N, D) = (1, N, D) = (1, 12, 4096)
        # Reshape to (B, emb_size, H, W, L)
        vit_out_reshaped = vit_out.permute(0, 2, 1)                                  # vit_out : (B, D, N) = (1, 4096, 12)
        vit_out_reshaped = vit_out_reshaped.contiguous().view(1, self.emb_size, h//6, w//6, l//6)   # vit_out : (B, D, 2, 2, 3)
        # Convolution to obtain again the same number of features of before the transformer
        vit_out_reshaped = self.conv_vit(vit_out_reshaped)                                    # vit_out : (1, 256, 12, 12, 18)

        d5 = self.dec_upconv_5(vit_out_reshaped) 
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
      
       
"""    
    # To summarize the complete model
from torchsummary import summary
model  = DeformationNetwork(2,1,[16, 32, 64, 128, 256],(256,256),256)
#print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
summary = summary(model, [(1, 192, 192, 300), (1, 192, 192, 300)], device='cuda')
"""
"""
# To summarize the complete model
from torchsummary import summary
model = ViTDefNetwork(2,1,[16, 32, 64, 128, 256],(192,192,300),4096)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

x = torch.randn(size=(1, 1, 192, 192, 300), dtype=torch.float32, device='cuda')
y = torch.randn(size=(1, 1, 192, 192, 300), dtype=torch.float32, device='cuda')
with torch.no_grad():
    x_0, y_0 = model(x, y)

print(f'Out: {x_0.shape}, Flow: {y_0.shape}')
"""

