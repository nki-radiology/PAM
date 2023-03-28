import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.SpatialTransformer import SpatialTransformer 
#from SpatialTransformer import SpatialTransformer # local 


"""
Convolution Class 
"""
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d   (in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU     (inplace=True),
            nn.Conv3d   (in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
        )

        self.identity = \
            nn.Conv3d   (in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding='same', bias=False)

    def forward(self, x):
        out = self.conv(x) + self.identity(x)
        return out 
        

"""
Encoder
"""
class Encoder(nn.Module):

    def __init__(self, img_size, filters, in_channels, out_channels) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2) # 96, 80
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2) # 48, 40
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2) # 24, 20
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2) # 12, 10
        self.Maxpool5 = nn.MaxPool3d(kernel_size=2, stride=2) # 6, 5

        self.Conv1    = Conv   (self.in_channels, self.filters[0])
        self.Conv2    = Conv   (self.filters[0],  self.filters[1])
        self.Conv3    = Conv   (self.filters[1],  self.filters[2])
        self.Conv4    = Conv   (self.filters[2],  self.filters[3])
        self.Conv5    = Conv   (self.filters[3],  self.filters[4])
        self.Conv6    = Conv   (self.filters[4],  self.filters[5])

        self.AvgPool  = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.Flatten  = nn.Flatten()
        self.Fc       = nn.Linear(self.filters[5], self.out_channels)
        self.Tanh     = nn.Tanh()

    def forward(self, image):

            x = self.Conv1(image)
            x = self.Maxpool1(x)

            x = self.Conv2(x)
            x = self.Maxpool2(x)

            x = self.Conv3(x)
            x = self.Maxpool3(x)

            x = self.Conv4(x)
            x = self.Maxpool4(x)

            x = self.Conv5(x)
            x = self.Maxpool5(x)

            x = self.Conv6(x)

            x = self.AvgPool(x)
            x = self.Flatten(x)
            x = self.Fc(x)
            #x = self.Tanh(x)

            return x


"""
Decoder
"""
class Decoder(nn.Module):

    def __init__(self, img_size, filters) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters

        self.DeConv6 = Conv       (self.filters[5], self.filters[4])
        self.UpConv6 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.DeConv5 = Conv       (self.filters[4], self.filters[3])
        self.UpConv5 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.DeConv4 = Conv       (self.filters[3], self.filters[2])
        self.UpConv4 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.DeConv3 = Conv       (self.filters[2], self.filters[1])
        self.UpConv3 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.DeConv2 = Conv       (self.filters[1], self.filters[0])
        self.UpConv2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.OutConv = nn.Conv3d(self.filters[0], 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, feature_maps):
                
        x  = self.DeConv6(feature_maps)
        x  = self.UpConv6(x) #12

        x  = self.DeConv5(x)
        x  = self.UpConv5(x) #24

        x  = self.DeConv4(x)
        x  = self.UpConv4(x) #48

        x  = self.DeConv3(x)
        x  = self.UpConv3(x) #96

        x  = self.DeConv2(x)
        x  = self.UpConv2(x) #192

        x = self.OutConv(x)

        return x


"""
Registration Network
"""
class PAMNetwork(nn.Module):    

    def __init__(self, img_size, filters) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters

        self.encoder        = Encoder(self.img_size, self.filters, in_channels=1, out_channels=1024)
        latent_dim          = self.encoder.out_channels

        # Affine Layers
        self.dense_w        = nn.Linear(in_features=latent_dim, out_features=9, bias=False)
        self.dense_b        = nn.Linear(in_features=latent_dim, out_features=3, bias=False)

        # Deformation Layers
        feature_maps_size   = [int(s/(2**5)) for s in self.img_size]
        elements            = np.prod(feature_maps_size) * self.filters[5]

        self.deflatten      = nn.Sequential(
            nn.Linear(latent_dim, elements),
            nn.ReLU()
        )

        self.decoder        = Decoder(self.img_size, self.filters)

        # Spatial layer
        self.spatial_layer  = SpatialTransformer(self.img_size)


    def forward(self, fixed, moving):

        # encoder
        z_fixed = self.encoder(fixed)
        z_moving = self.encoder(moving)
        z = z_fixed - z_moving

        # compute affine transform
        W = self.dense_w(z).view(-1, 3, 3)
        b = self.dense_b(z).view(-1, 3)

        tA = torch.cat((W, b.unsqueeze(dim=1)), dim=1)
        tA = tA.view(-1, 3, 4)
        tA = F.affine_grid(tA, moving.size(), align_corners=False)
        tA = tA.permute(0, 4, 1, 2, 3)
        
        # compute deformation field
        x  = self.deflatten(z)

        s = [int(s/(2**5)) for s in self.img_size]
        s = (z.shape[0], self.filters[5], s[0], s[1], s[2])
        x = torch.reshape(x, s)

        tD = self.decoder(x)

        # apply transforms
        # wD = self.spatial_layer(moving, tA+tD)
        wA = self.spatial_layer(moving, tA)
        wD = self.spatial_layer(moving, tD)

        return tA, wA, tD, wD


        
"""
# To summarize the complete model
from torchsummary import summary
img_size = [192, 192, 160]
filters = [8, 8, 16, 32, 64, 128]
model  = PAMNetwork(img_size, filters)
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
summary = summary(model, [(1, 192, 192, 160), (1, 192, 192, 160)], device='cuda')
"""