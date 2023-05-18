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
    def __init__(self, in_ch, out_ch, downsample=False):
        super(Conv, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.downsample = downsample

        s = 2 if downsample else 1
        p = 'valid' if downsample else 'same'

        self.conv0      = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=s, padding=p, bias=False)

        self.conv1      = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding='same', bias=False)
        self.conv2      = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv3      = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=s, padding=p, bias=False)

        self.relu1      = nn.LeakyReLU(inplace=True)
        self.relu2      = nn.LeakyReLU(inplace=True)
        self.relu3      = nn.LeakyReLU(inplace=True)

        self.gnorm1     = nn.GroupNorm(num_groups=4, num_channels=out_ch)
        self.gnorm2     = nn.GroupNorm(num_groups=4, num_channels=out_ch)
        self.gnorm3     = nn.GroupNorm(num_groups=4, num_channels=out_ch)

    def forward(self, x):

        out = self.conv1(x)
        out = self.gnorm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.gnorm2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.gnorm3(out)

        if self.downsample or (self.in_ch != self.out_ch):
            x = self.conv0(x)

        out = out + x
        out = self.relu3(out)

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

        self.Conv1    = Conv   (self.in_channels, self.filters[0], downsample=True)
        self.Conv2    = Conv   (self.filters[0],  self.filters[1], downsample=True)
        self.Conv3    = Conv   (self.filters[1],  self.filters[2], downsample=True)
        self.Conv4    = Conv   (self.filters[2],  self.filters[3], downsample=True)
        self.Conv5    = Conv   (self.filters[3],  self.filters[4], downsample=True)
        self.Conv6    = Conv   (self.filters[4],  self.filters[5])

        self.AvgPool  = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.Flatten  = nn.Flatten()
        self.Fc       = nn.Linear(self.filters[5], self.out_channels, bias=False)

    def forward(self, image):

            x = self.Conv1(image)
            x = self.Conv2(x)
            x = self.Conv3(x)
            x = self.Conv4(x)
            x = self.Conv5(x)
            x = self.Conv6(x)

            x = self.AvgPool(x)
            x = self.Flatten(x)
            x = self.Fc(x)

            return x

"""
Decoder
"""
class AffineDecoder(nn.Module):

    def __init__(self, image_dim, latent_dim) -> None:
        super().__init__()
        self.image_dim = image_dim
        self.latent_dim = latent_dim

        self.dense_w = nn.Linear(in_features=self.latent_dim, out_features=9, bias=False)
        self.dense_b = nn.Linear(in_features=self.latent_dim, out_features=3, bias=False)

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
    

class ElasticDecoder(nn.Module):

    def __init__(self, img_size, filters, latent_dim) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters
        self.latent_dim = latent_dim

        feature_maps_size   = [int(s/(2**5)) for s in self.img_size]
        elements            = np.prod(feature_maps_size) * self.filters[5]

        self.deflatten      = nn.Sequential(
            nn.Linear(latent_dim, elements, bias=False),
            nn.LeakyReLU()
        )

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

    def forward(self, z):
                
        x = self.deflatten(z)

        s = [int(s/(2**5)) for s in self.img_size]
        s = (z.shape[0], self.filters[5], s[0], s[1], s[2])
        feature_maps = torch.reshape(x, s)
            
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

    def __init__(self, img_size, filters, latent_dim) -> None:
        super().__init__()
        self.img_size = img_size
        self.filters = filters
        self.latent_dim = latent_dim

        self.encoder            = Encoder(self.img_size, self.filters, in_channels=1, out_channels=self.latent_dim)
        self.decoder_affine     = AffineDecoder(self.img_size, self.latent_dim*2)
        self.decoder_deform     = ElasticDecoder(self.img_size, self.filters, self.latent_dim*2)

        self.spatial_layer      = SpatialTransformer(self.img_size)


    def forward(self, fixed, moving, compute_residuals=False):

        def compute_t(fixed, moving, encoder, decoder):
            # repeated operation 
            z_fixed = encoder(fixed)
            z_moving = encoder(moving if w_ is None else w_)

            z_diff = z_fixed - z_moving
            z = torch.concat((z_fixed, z_diff), dim=1)

            t = decoder(z)
            return z, t

        # registrations
        zA, tA = compute_t(fixed, moving, self.encoder, self.decoder_affine)
        wA = self.spatial_layer(moving, tA)
        zD, tD = compute_t(fixed, wA, self.encoder, self.decoder_deform)
        wD = self.spatial_layer(moving, tA + tD)

        # residuals
        residual = None
        if compute_residuals:
            _, (z, _, _), _  = self.forward(fixed, wD, compute_residuals=False)
            residual = z[-self.latent_dim:]

        return (zA, tA, wA), (zD, tD, wD), residual

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