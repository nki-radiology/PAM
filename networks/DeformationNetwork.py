import torch
import torch.nn as nn
from   networks.SpatialTransformer import SpatialTransformer



"""
Convolution Class for the U-Net generator
"""
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d   (in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm( num_groups=8,     num_channels=out_ch),
            nn.ReLU     (inplace=True),
            nn.Conv3d   (in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm( num_groups=8,      num_channels=out_ch),
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
            nn.GroupNorm(num_groups =8,      num_channels=out_ch),
            nn.ReLU     (inplace=True)
        )

    def forward(self, x):
        out = self.up(x)
        return out

"""
U-Net Generator class
"""
class DeformationNetwork(nn.Module):

    def __init__(self, filters, img_dim):
        super(DeformationNetwork, self).__init__()
        self.filters = filters       # [16, 32, 64, 128, 256]
        self.img_dim = img_dim

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2) # 96, 80
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2) # 48, 40
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2) # 24, 20
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2) # 12, 10

        self.Conv1    = Conv   (1,               self.filters[0])
        self.Conv2    = Conv   (self.filters[0], self.filters[1])
        self.Conv3    = Conv   (self.filters[1], self.filters[2])
        self.Conv4    = Conv   (self.filters[2], self.filters[3])
        self.Conv5    = Conv   (self.filters[3], self.filters[4])

        self.AvgPool  = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.Fc       = nn.Linear(self.filters[4], self.filters[4])
        self.FcAct    = nn.Tanh 

        self.encoder = nn.Sequential(
            self.Conv1, self.Maxpool1,
            self.Conv2, self.Maxpool2,
            self.Conv3, self.Maxpool3,
            self.Conv4, self.Maxpool4,
            self.Conv5, 
            self.AvgPool,
            self.Fc #, 
            #self.FcAct
        )        

        self.Up5      = Up_Conv(self.filters[4], self.filters[3])
        self.Up_conv5 = Conv   (self.filters[4], self.filters[3])

        self.Up4      = Up_Conv(self.filters[3], self.filters[2])
        self.Up_conv4 = Conv   (self.filters[3], self.filters[2])

        self.Up3      = Up_Conv(self.filters[2], self.filters[1])
        self.Up_conv3 = Conv   (self.filters[2], self.filters[1])

        self.Up2      = Up_Conv(self.filters[1], self.filters[0])
        self.Up_conv2 = Conv   (self.filters[1], self.filters[0])

        self.Conv     = nn.Conv3d(self.filters[0], 3, kernel_size=1, stride=1, padding=0, bias=False)

        self.spat_trs = SpatialTransformer(self.img_dim)  #((192, 192, 160))


    def forward(self, fixed, moving):

        z_fixed     = self.encoder(fixed)
        z_moving    = self.encoder(moving)

        z           = z_fixed - z_moving

        # tiling the feature vector to feature map
        z = z[..., None, None, None]
        s = [int(s/(2**4)) for s in self.img_dim]
        z = z.tile((1, 1, s[0], s[1], s[2]))

        d5  = self.Up5     (z)
        d5  = self.Up_conv5(d5)

        d4  = self.Up4     (d5)
        d4  = self.Up_conv4(d4)

        d3  = self.Up3     (d4)
        d3  = self.Up_conv3(d3)

        d2  = self.Up2     (d3)
        d2  = self.Up_conv2(d2)

        transformation = self.Conv(d2)

        # Spatial Transform
        registered_img = self.spat_trs(moving, transformation)

        return transformation, registered_img

"""
# To summarize the complete model
from torchsummary import summary
model  = DeformationNetwork()
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
summary = summary(model, [(1, 192, 192, 160), (1, 192, 192, 160)], device='cuda')
"""