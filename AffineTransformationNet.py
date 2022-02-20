import torch
import numpy as np
import torch as nn
from config import args_at
from torch.nn import GroupNorm
import torch.nn.functional as F
from layers import encoding_relu
from layers import decoding_relu


class AffineTransformation(nn.Module):
    def __init__(self, dim=1, in_nf = 16, img_size=512):
        super(AffineTransformation).__init__()
        self.dim  = dim
        self.in_nf= in_nf
        self.n_fil= [self.in_nf, self.in_nf * 2,
                     self.in_nf * 4, self.in_nf * 8, self.in_nf * 16]

        self.enc1 = encoding_relu(2, self.n_fil[0], 3, 2, 3, 8)      # 01 x 16
        self.enc2 = encoding_relu(self.n_fil[0], self.n_fil[1], 3, 2, 3, 8) # 16 x 32
        self.enc3 = encoding_relu(self.n_fil[1], self.n_fil[2], 3, 2, 3, 8) # 32 x 64
        self.enc4 = encoding_relu(self.n_fil[2], self.n_fil[3], 3, 2, 3, 8) # 64 x 128
        self.enc5 = encoding_relu(self.n_fil[3], self.n_fil[4], 3, 2, 3, 8) # 128x 256

        # self.avp1 = nn.AvgPool3d((1, 1))
        # Considering that image shape is like (img_size, img_size, img_size)
        self.size = img_size // (self.in_nf * 4)
        self.dense= nn.Sequential(
            nn.Linear(512 * self.size**dim, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 6*(dim - 1)),
            # nn.ReLU(True)
        )

        # Initialization of  weights and bias using the identity transformation
        self.dense[-1].weight.data.zero()
        if dim == 3:
            self.dense[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.dense[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, fixed, moving):
        concat_img = torch.cat((fixed, moving), dim=1)  # 02x512x512
        encoding1  = self.enc1(concat_img)              # 16x256x256
        encoding2  = self.enc2(encoding1)               # 32x128x128
        encoding3  = self.enc3(encoding2)               # 64x64x64
        encoding4  = self.enc4(encoding3)               # 128x32x32
        encoding5  = self.enc5(encoding4)               # 256x16x16

        # Affine Transformation: y = Wx + b
        x = encoding5.view(-1, 256 * self.size ** self.dim)
        if self.dim == 3:
            w = self.dense(x).view(-1, 3, 4)
        else:
            w = self.dense(x).view(-1, 2, 3)

        grid = F.affine_grid(w, moving.size(), align_corners=False)

        if self.dim == 2:
            grid = grid.permute(0, 3, 1, 2)  # batch x 2 x 512 x 512
        else:
            grid = grid.permute(0, 4, 1, 2, 3)







