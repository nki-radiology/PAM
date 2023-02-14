import torch
import torch.nn as nn
import os
from   general_config              import deformation, visiontransformer
from   networks.SpatialTransformer import SpatialTransformer  
from   networks.DeformationNetwork import DeformationNetwork
from   networks.ViTransformer      import *                
from   networks.layers    import *    

if deformation.skip_choice == 'yes':
    skip_choice = 1 #--Unet
elif deformation.skip_choice == 'no':
    skip_choice = 0 #--Autoencoder


class ViTDefNetwork(DeformationNetwork):
    
    def __init__(self, in_channels, out_channels, filters, img_dim, emb_size, num_heads, num_layers):
        super().__init__(in_channels, out_channels, filters, img_dim)
        
        self.emb_size = visiontransformer.emb_size              # 4096 if big_net, 1024 if small_net   
        self.num_heads = visiontransformer.ViT_heads            # default = 16
        self.num_layers = visiontransformer.ViT_layers          # default = 12
              
        self.convnet['vit'] = Transformer(emb_size    = self.emb_size, 
                               in_channels = self.filters[4], 
                               patch_size  = 6,
                               num_heads   = self.num_heads,
                               num_layers  = self.num_layers
                              ) 
        self.convnet['conv_vit'] = nn.Sequential(
            nn.Upsample (scale_factor = 6, mode = 'trilinear'),
            nn.Conv3d   (in_channels  = self.emb_size, out_channels = self.filters[4], kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(num_groups   = GN_number, num_channels = self.filters[4]),
            nn.ReLU     (inplace      = True))
                           
    def forward(self, fixed, moving): 

        x = torch.cat((fixed, moving), dim=1) 
        e1 = self.convnet['encoder_conv_1'](x)

        e2 = self.convnet['encoder_max_1'](e1)
        e2 = self.convnet['encoder_conv_2'](e2)

        e3 = self.convnet['encoder_max_2'](e2)
        e3 = self.convnet['encoder_conv_3'](e3)

        e4 = self.convnet['encoder_max_3'](e3) 
        e4 = self.convnet['encoder_conv_4'](e4)

        e5 = self.convnet['encoder_max_4'](e4)
        e5 = self.convnet['encoder_conv_5'](e5) 

        # Application of the Transformer to the feature-map of dimensions (1, 256, 12, 12, 18)
        vit_in  = torch.squeeze(e5)                                                             # vit_in : (256, 12, 12, 18)
        h, w, l = vit_in.size(1), vit_in.size(2), vit_in.size(3)
        vit_out = self.convnet['vit'](e5)                                                       # vit_out : (B, N, D) = (1, N, D) = (1, 12, 4096)
        vit_out = vit_out.permute(0, 2, 1)                                                      # vit_out : (B, D, N) = (1, 4096, 12)
        vit_out = vit_out.contiguous().view(1, self.emb_size, h//6, w//6, l//6)                 # vit_out : (B, D, 2, 2, 3)
        vit_out = self.convnet['conv_vit'](vit_out)                                             # vit_out : (1, 256, 12, 12, 18)

        d5 = self.convnet['decoder_upconv_1'](vit_out)         

        d5 = autoexpand(e4, d5)
        if skip_choice == 1:
            d5  = torch.cat((e4, d5), dim=1)        
        d5 = self.convnet['decoder_conv_1'](d5)
        
        d4 = self.convnet['decoder_upconv_2'](d5)
        d4 = autoexpand(e3, d4)
        if skip_choice == 1:
            d4  = torch.cat((e3, d4), dim=1)
        d4 = self.convnet['decoder_conv_2'](d4)
        
        d3 = self.convnet['decoder_upconv_3'](d4)
        d3 = autoexpand(e2, d3)
        if skip_choice == 1:
            d3  = torch.cat((e2, d3), dim=1)
        d3 = self.convnet['decoder_conv_3'](d3)
        
        d2 = self.convnet['decoder_upconv_4'](d3)
        d2 = autoexpand(e1, d2)
        if skip_choice == 1:
            d2  = torch.cat((e1, d2), dim=1)
        d2 = self.convnet['decoder_conv_4'](d2)
        
        deformation_field = self.convnet['transformation'](d2) 
        registered_img = self.spat_trs(moving, deformation_field)
               
        return deformation_field, registered_img


"""model = ViTDefNetwork(2,3,[16, 32, 64, 128, 256],(192,192,300),4096, 16, 12)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = torch.randn(size=(1, 1, 192, 192, 300), dtype=torch.float32, device='cuda')
y = torch.randn(size=(1, 1, 192, 192, 300), dtype=torch.float32, device='cuda')
with torch.no_grad():
    x_0, y_0 = model(x, y)

print(f'Out: {x_0.shape}, Flow: {y_0.shape}')
"""

"""# To summarize the complete model
from torchsummary import summary
model = ViTDefNetwork(2,3,[16, 32, 64, 128, 256],(192,192,300),4096, 16, 12)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary = summary(model, [(1, 192, 192, 300), (1, 192, 192, 300)], device='cuda')
"""
