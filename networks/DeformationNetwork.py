import torch
import torch.nn as nn
from   networks.SpatialTransformer import SpatialTransformer
#from   SpatialTransformer import SpatialTransformer
from   general_config              import deformation
from   networks.layers             import *   
import os

if deformation.skip_choice == 'yes':
    skip_choice = 1 #--Unet
elif deformation.skip_choice == 'no':
    skip_choice = 0 #--Autoencoder


class DeformationNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, filters, img_dim):
        super(DeformationNetwork, self).__init__()
        self.in_ch   = in_channels   # 2
        self.out_ch  = out_channels  # 3
        
        self.filters = filters
        self.img_dim = img_dim       # (192,192,300)

        seq = [self.in_ch] + self.filters
        self.convnet = nn.ModuleDict()

        for i, (j, k) in enumerate(zip(seq, seq[1:])):
            self.convnet['encoder_conv_' + str(i+1)] = Conv(j, k)

        for i, _ in enumerate(seq):
            self.convnet['encoder_max_' + str(i+1)] = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        seq = list(reversed(self.filters))

        for i, (j, k) in enumerate(zip(seq, seq[1:])):
            self.convnet['decoder_upconv_' + str(i+1)] = Up_Conv(j, k)

        if skip_choice == 1: 
            for i, (j, k) in enumerate(zip(seq, seq[1:])):
                self.convnet['decoder_conv_' + str(i+1)] = Conv(j, k)
        else: 
            for i, (j, k) in enumerate(zip(seq, seq[1:])):
                self.convnet['decoder_conv_' + str(i+1)] = Conv_noskip(k)
                

        self.convnet['transformation'] = nn.Conv3d(self.filters[0], self.out_ch, kernel_size=1, stride=1, padding=0, bias=False)  

        self.spat_trs = SpatialTransformer(self.img_dim)       


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
        
        d5 = self.convnet['decoder_upconv_1'](e5)         

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

    
"""# To summarize the complete model
from torchsummary import summary
model = DeformationNetwork(2,3,[16, 32, 64, 128, 256],(192,192,300))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

x = torch.randn(size=(1, 1, 192, 192, 300), dtype=torch.float32, device=device)
y = torch.randn(size=(1, 1, 192, 192, 300), dtype=torch.float32, device=device)
with torch.no_grad():
    transformation0, registered0 = model(x, y)

print(f'Out: {transformation0.shape}, Flow: {registered0.shape}')
"""


"""# To summarize the complete model
from torchsummary import summary
model  = DeformationNetwork(2,3,[16, 32, 64, 128, 256],(192,192,300))
print(model)
device = torch.device("cuda:0")
model  = model.to(device)
summary = summary(model, [(1, 192, 192, 300), (1, 192, 192, 300)], device='cuda')
"""