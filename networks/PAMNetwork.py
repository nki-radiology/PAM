import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.SpatialTransformer import SpatialTransformer 
#from SpatialTransformer import SpatialTransformer # local 


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv      = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding='same', bias=False)
        self.gnorm     = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.relu      = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.gnorm(out)
        out = self.relu(out)

        return out


class ResConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv0      = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding='same', bias=False)

        self.conv1      = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding='same', bias=False)
        self.conv2      = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv3      = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding='same', bias=False)

        self.relu1      = nn.LeakyReLU(inplace=True)
        self.relu2      = nn.LeakyReLU(inplace=True)
        self.relu3      = nn.LeakyReLU(inplace=True)

        self.gnorm1     = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.gnorm2     = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.gnorm3     = nn.GroupNorm(num_groups=8, num_channels=out_ch)

    def forward(self, x):

        out = self.conv1(x)
        out = self.gnorm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.gnorm2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.gnorm3(out)

        if self.in_ch != self.out_ch:
            x = self.conv0(x)

        out = out + x
        out = self.relu3(out)

        return out 


class Encoder(nn.Module):

    def __init__(self, img_size, filters, in_channels, out_channels, flatten=False) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flatten = flatten

        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1    = ResConv   (self.in_channels, self.filters[0])
        self.Conv2    = ResConv   (self.filters[0],  self.filters[1])
        self.Conv3    = ResConv   (self.filters[1],  self.filters[2])
        self.Conv4    = ResConv   (self.filters[2],  self.filters[3])
        self.Conv5    = ResConv   (self.filters[3],  self.filters[4])
        self.Conv6    = ResConv   (self.filters[4],  self.filters[5])

        if self.flatten:
            self.AvgPool  = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
            self.Flatten  = nn.Flatten()
            self.Fc       = nn.Linear(self.filters[5], self.out_channels, bias=False)

    def forward(self, image):

            x = self.Conv1(image)
            x = self.MaxPool(x)

            x = self.Conv2(x)
            x = self.MaxPool(x)

            x = self.Conv3(x)
            x = self.MaxPool(x)

            x = self.Conv4(x)
            x = self.MaxPool(x)

            x = self.Conv5(x)
            x = self.MaxPool(x)
            
            x = self.Conv6(x)

            if self.flatten:
                x = self.AvgPool(x)
                x = self.Flatten(x)
                x = self.Fc(x)

            return x


class Decoder(nn.Module):

    def __init__(self, img_size, filters, in_channels, out_channels, deflatten=False) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deflatten = deflatten

        if self.deflatten:
            feature_maps_size       = [int(s/(2**5)) for s in self.img_size]
            elements                = np.prod(feature_maps_size) * self.filters[5]

            self.deflatten_layer    = nn.Sequential(
                nn.Linear(in_channels, elements, bias=False),
                nn.LeakyReLU()
            )

        self.UpSample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.DeConv6 = Conv       (self.filters[5], self.filters[4])
        self.DeConv5 = Conv       (self.filters[4], self.filters[3])
        self.DeConv4 = Conv       (self.filters[3], self.filters[2])
        self.DeConv3 = Conv       (self.filters[2], self.filters[1])
        self.DeConv2 = Conv       (self.filters[1], self.filters[0])

        self.OutConv = nn.Conv3d(self.filters[0], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, z):
                
        if self.deflatten:
            x = self.deflatten_layer(z)

            s = [int(s/(2**5)) for s in self.img_size]
            s = (z.shape[0], self.filters[5], s[0], s[1], s[2])
            z = torch.reshape(x, s)
                
        x  = self.DeConv6(z)
        x  = self.UpSample(x) #12

        x  = self.DeConv5(x)
        x  = self.UpSample(x) #24

        x  = self.DeConv4(x)
        x  = self.UpSample(x) #48

        x  = self.DeConv3(x)
        x  = self.UpSample(x) #96

        x  = self.DeConv2(x)
        x  = self.UpSample(x) #192

        x = self.OutConv(x)

        return x


class AffineDecoder(nn.Module):

    def __init__(self, image_dim, in_channels) -> None:
        super().__init__()
        self.image_dim = image_dim
        self.in_channels = in_channels

        self.dense_w = nn.Linear(in_features=self.in_channels, out_features=9, bias=False)
        self.dense_b = nn.Linear(in_features=self.in_channels, out_features=3, bias=False)

    def forward(self, z):            
        # compute affine transform
        W = self.dense_w(z).view(-1, 3, 3)
        b = self.dense_b(z).view(-1, 3)

        tA = torch.cat((W, b.unsqueeze(dim=1)), dim=1)
        tA = tA.view(-1, 3, 4)
        target_shape = torch.Size((z.shape[0], 1, *self.image_dim))
        tA = F.affine_grid(tA, target_shape, align_corners=False)
        tA = tA.permute(0, 4, 1, 2, 3)

        return tA
    

class UNet(nn.Module):

    def __init__(self, img_size, filters, in_channels, out_channels) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = Encoder(self.img_size, self.filters, in_channels=self.in_channels, out_channels=self.filters[-1], flatten=False)

        self.UpSample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.DeConv6 = Conv       (self.filters[5]+self.filters[4], self.filters[4])
        self.DeConv5 = Conv       (self.filters[4]+self.filters[3], self.filters[3])
        self.DeConv4 = Conv       (self.filters[3]+self.filters[2], self.filters[2])
        self.DeConv3 = Conv       (self.filters[2]+self.filters[1], self.filters[1])
        self.DeConv2 = Conv       (self.filters[1]+self.filters[0], self.filters[0])

        self.OutConv = nn.Conv3d(self.filters[0], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, image):


        x1  = self.encoder.Conv1(image)
        x   = self.encoder.MaxPool(x1)

        x2  = self.encoder.Conv2(x)
        x   = self.encoder.MaxPool(x2)

        x3  = self.encoder.Conv3(x)
        x   = self.encoder.MaxPool(x3)

        x4  = self.encoder.Conv4(x)
        x   = self.encoder.MaxPool(x4)

        x5  = self.encoder.Conv5(x)
        x   = self.encoder.MaxPool(x5)

        x   = self.encoder.Conv6(x)

        x   = self.UpSample(x)
        x   = torch.cat((x, x5), dim=1)
        x   = self.DeConv6(x)

        x   = self.UpSample(x)
        x   = torch.cat((x, x4), dim=1)
        x   = self.DeConv5(x)

        x   = self.UpSample(x)
        x   = torch.cat((x, x3), dim=1)
        x   = self.DeConv4(x)

        x   = self.UpSample(x)
        x   = torch.cat((x, x2), dim=1)
        x   = self.DeConv3(x)

        x   = self.UpSample(x)
        x   = torch.cat((x, x1), dim=1)
        x   = self.DeConv2(x)

        x   = self.OutConv(x)

        return x
    

class RegistrationNetwork(nn.Module):

    def __init__(self, img_size, filters) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters

        self.encoder            = Encoder(self.img_size, self.filters, in_channels=2, out_channels=self.filters[-1], flatten=True)
        self.decoder_affine     = AffineDecoder(self.img_size, in_channels=self.filters[-1])
        self.unet               = UNet(self.img_size, self.filters, in_channels=2, out_channels=3)

        self.spatial_layer      = SpatialTransformer(self.img_size)


    def forward(self, fixed, moving):
        # compute affine transform
        z = self.encoder(torch.cat((fixed, moving), dim=1))
        tA = self.decoder_affine(z)
        wA = self.spatial_layer(moving, tA)

        # compute deformation field
        tD = self.unet(torch.cat((fixed, wA), dim=1))
        wD = self.spatial_layer(moving, tA + tD)

        return (wA, wD), (tA, tD)
    

class RegistrationNetworkV2(nn.Module):

    def __init__(self, img_size, filters) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters

        self.encoder            = Encoder(self.img_size, self.filters, in_channels=1, out_channels=self.filters[-1], flatten=True)
        self.decoder_affine     = AffineDecoder(self.img_size, in_channels=self.filters[-1]*2)
        self.unet               = UNet(self.img_size, self.filters, in_channels=2, out_channels=3)

        self.spatial_layer      = SpatialTransformer(self.img_size)


    def forward(self, fixed, moving):
        # compute affine transform
        z_fixed     = self.encoder(fixed)
        z_moving    = self.encoder(moving)
        z           = torch.cat((z_fixed, z_moving), dim=1)
        tA          = self.decoder_affine(z)
        wA          = self.spatial_layer(moving, tA)

        # compute deformation field
        tD          = self.unet(torch.cat((fixed, wA), dim=1))
        wD          = self.spatial_layer(wA, tD)

        return (wA, wD), (tA, tD)
        

class SegmentationNetwork(nn.Module):

    def __init__(self, img_size, filters, n_classes) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters
        self.n_classes = n_classes

        self.unet       = UNet(self.img_size, self.filters, in_channels=1, out_channels=n_classes)
        self.softmax    = nn.Softmax(dim=1)


    def forward(self, image):
        x = self.unet(image)
        x = self.softmax(x)

        return x


class StudentNetwork(nn.Module):
    
        def __init__(self, img_size, filters, n_classes, latent_dim) -> None:
            super().__init__()
            self.img_size   = img_size
            self.n_classes  = n_classes
            self.filters    = filters
            self.latent_dim = latent_dim
    
            self.encoder        = Encoder(self.img_size, self.filters, in_channels=1, out_channels=self.latent_dim, flatten=True)

            # registration pathway
            self.decoder_reg    = Decoder(self.img_size, self.filters, in_channels=self.latent_dim*2, out_channels=3, deflatten=True)
            self.spatial_layer  = SpatialTransformer(self.img_size)

            # segmentation pathway
            self.decoder_seg    = Decoder(self.img_size, self.filters, in_channels=self.latent_dim, out_channels=self.n_classes, deflatten=True)
            self.softmax        = nn.Softmax(dim=1)

    
        def forward(self, fixed, moving, return_embedding=False):    
            # encoding
            z_fixed     = self.encoder(fixed)
            z_moving    = self.encoder(moving)

            z_diff      = z_moving - z_fixed
            z           = torch.concat((z_moving, z_diff), dim=1)

            if return_embedding:
                return z_fixed, z_moving, z_diff

            # registration pathway
            t = self.decoder_reg(z)
            w = self.spatial_layer(moving, t)

            # segmentation pathway
            s_fixed = self.decoder_seg(z_fixed)
            s_fixed = self.softmax(s_fixed)

            s_moving = self.decoder_seg(z_moving)
            s_moving = self.softmax(s_moving)
            
            return (w, t), (s_fixed, s_moving)
        


"""
# To summarize the complete model
from torchsummary import summary
img_size = [192, 192, 160]
filters = [8, 8, 16, 32, 64, 128]
model  = PAMNetwork(img_size, filters, latent_dim=64)
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
summary = summary(model, [(1, 192, 192, 160), (1, 192, 192, 160)], device='cuda')
"""