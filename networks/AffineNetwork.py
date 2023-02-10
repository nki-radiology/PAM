import torch
import torch.nn            as nn
import torch.nn.functional as F
from networks.SpatialTransformer import SpatialTransformer


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


class AffineNetwork(nn.Module):

    def __init__(self, in_channels, img_dim):
        super(AffineNetwork, self).__init__()
        self.in_ch   = in_channels # 2
        
        filters = [32, 64, 128, 256, 512]
        self.filters = filters
        self.img_dim = img_dim     # (192,192,300)

        self.conv1 = S_Conv(self.in_ch,      self.filters[0])
        self.conv2 = S_Conv(self.filters[0], self.filters[1])
        self.conv3 = S_Conv(self.filters[1], self.filters[2])
        self.conv4 = S_Conv(self.filters[2], self.filters[3])
        self.conv5 = S_Conv(self.filters[3], self.filters[4])

        # Last Layer
        self.glAvg = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dense = nn.Linear(in_features=self.filters[4], out_features=1024, bias=False)
        self.lfact = nn.ReLU(inplace=True)

        # Affine Layers
        self.den_w = nn.Linear(in_features=1024, out_features=9, bias=False)
        self.den_b = nn.Linear(in_features=1024, out_features=3, bias=False)

        # Spatial Transformer
        self.sp_t = SpatialTransformer(self.img_dim)


    def forward(self, fixed, moving):

        x = torch.cat((fixed, moving), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Last Layer
        x = self.glAvg(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = self.lfact(x)

        # Affine Matrix
        W = self.den_w(x).view(-1, 3, 3)
        b = self.den_b(x).view(-1, 3)
        I = torch.eye(3, dtype=torch.float32, requires_grad=True, device="cuda")

        # Input for the Spatial Transform Network
        transformation = torch.cat((W, b.unsqueeze(dim=1)), dim=1)
        transformation = transformation.view(-1, 3, 4)
        transformation = F.affine_grid(transformation, moving.size(), align_corners=False)
        transformed    = transformation.permute(0, 4, 1, 2, 3)

        # Spatial transform
        registered     = self.sp_t(moving, transformed)

        return transformation, registered

"""    
# To summarize the complete model
from torchsummary import summary
model  = AffineNetwork(2,(192,192,300))
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
summary = summary(model, [(1, 192, 192, 300), (1, 192, 192, 300)], device='cuda')
"""