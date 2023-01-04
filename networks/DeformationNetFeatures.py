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
            nn.Conv3d   (in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False), ### Here!
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

    def __init__(self, in_channels, out_channels, filters, img_dim):
        super(DeformationNetwork, self).__init__()
        self.in_ch   = in_channels   # 2
        self.out_ch  = out_channels  # 1
        self.filters = filters       # [16, 32, 64, 128, 256]
        self.img_dim = img_dim

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1    = Conv   (self.in_ch,      self.filters[0])
        self.Conv2    = Conv   (self.filters[0], self.filters[1])
        self.Conv3    = Conv   (self.filters[1], self.filters[2])
        self.Conv4    = Conv   (self.filters[2], self.filters[3])
        self.Conv5    = Conv   (self.filters[3], self.filters[4])
       

        self.Up5      = Up_Conv(self.filters[4], self.filters[3])
        self.Up_conv5 = Conv   (self.filters[4], self.filters[3])

        self.Up4      = Up_Conv(self.filters[3], self.filters[2])
        self.Up_conv4 = Conv   (self.filters[3], self.filters[2])

        self.Up3      = Up_Conv(self.filters[2], self.filters[1])
        self.Up_conv3 = Conv   (self.filters[2], self.filters[1])

        self.Up2      = Up_Conv(self.filters[1], self.filters[0])
        self.Up_conv2 = Conv   (self.filters[1], self.filters[0])

        self.Conv     = nn.Conv3d(self.filters[0], self.out_ch, kernel_size=1, stride=1, padding=0, bias=False)

        self.spat_trs = SpatialTransformer(self.img_dim)  #((192, 192, 160))


    def forward_init(self, fixed, moving):

        x = torch.cat((fixed, moving), dim=1)
        e1  = self.Conv1(x)

        e2  = self.Maxpool1(e1)
        e2  = self.Conv2   (e2)

        e3  = self.Maxpool2(e2)
        e3  = self.Conv3   (e3)

        e4  = self.Maxpool3(e3)
        e4  = self.Conv4   (e4)

        e5  = self.Maxpool4(e4)
        e5  = self.Conv5   (e5)
        


        d5  = self.Up5     (e5)
        d5  = torch.cat((e4, d5), dim=1)
        d5  = self.Up_conv5(d5)

        d4  = self.Up4     (d5)
        d4  = torch.cat((e3, d4), dim=1)
        d4  = self.Up_conv4(d4)

        d3  = self.Up3     (d4)
        d3  = torch.cat((e2, d3), dim=1)
        d3  = self.Up_conv3(d3)

        d2  = self.Up2     (d3)
        d2  = torch.cat((e1, d2), dim=1)
        d2  = self.Up_conv2(d2)

        transformation = self.Conv(d2)

        # Spatial Transform
        registered_img = self.spat_trs(moving, transformation)

        return transformation, registered_img


    def forward(self, fixed, moving):

        x = torch.cat((fixed, moving), dim=1)
        e1  = self.Conv1(x)

        e2  = self.Maxpool1(e1)
        e2  = self.Conv2   (e2)

        e3  = self.Maxpool2(e2)
        e3  = self.Conv3   (e3)

        e4  = self.Maxpool3(e3)
        e4  = self.Conv4   (e4)

        e5  = self.Maxpool4(e4)
        e6  = self.Conv5   (e5) #e5  = self.Conv5   (e5)
        #ls  = self.Conv5ls (e5)


        d5  = self.Up5     (e6) #(e5)
        d5  = torch.cat((e4, d5), dim=1)
        d5  = self.Up_conv5(d5)

        d4  = self.Up4     (d5)
        d4  = torch.cat((e3, d4), dim=1)
        d4  = self.Up_conv4(d4)

        d3  = self.Up3     (d4)
        d3  = torch.cat((e2, d3), dim=1)
        d3  = self.Up_conv3(d3)

        d2  = self.Up2     (d3)
        d2  = torch.cat((e1, d2), dim=1)
        d2  = self.Up_conv2(d2)

        transformation = self.Conv(d2)

        # Spatial Transform
        registered_img = self.spat_trs(moving, transformation)

        return e4, e6, d5, transformation, registered_img  # ls

"""
# To summarize the complete model
from torchsummary import summary
model  = DeformationNetwork()
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
summary = summary(model, [(1, 192, 192, 160), (1, 192, 192, 160)], device='cuda')
"""