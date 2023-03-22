import torch
import torch.nn as nn
import sys


"""
Simple convolution class
"""
class S_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(S_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d   (in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8,      num_channels=out_ch),
            nn.ReLU     (inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class DiscriminatorNetwork(nn.Module):

    def __init__(self, img_dim, filters):
        super(DiscriminatorNetwork, self).__init__()
        self.img_size = img_dim
        self.filters = filters     # [8, 16, 32, 64, 128, 256, 512]

        self.conv1 = S_Conv(1,               self.filters[0])
        self.conv2 = S_Conv(self.filters[0], self.filters[1])
        self.conv3 = S_Conv(self.filters[1], self.filters[2])
        self.conv4 = S_Conv(self.filters[2], self.filters[3])
        self.conv5 = S_Conv(self.filters[3], self.filters[4])
        self.conv6 = S_Conv(self.filters[4], self.filters[5])
        self.conv7 = S_Conv(self.filters[5], self.filters[6])


        # Last Layer
        self.h     = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dense = nn.Linear(in_features=self.filters[6], out_features=1, bias=False)
        self.act   = nn.Sigmoid()


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        f_x = x = self.conv7(x)

        # Last Layer
        x = self.h(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = self.act(x)

        return x, f_x

"""
# To summarize the complete model
from torchsummary import summary
model  = DiscriminatorNetwork()
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
summary = summary(model, (1, 192, 192, 160), device='cuda')
"""