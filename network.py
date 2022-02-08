import numpy as np
import torch as nn
from torch.nn import GroupNorm
from config import args_at
from layers import encoding_relu
from layers import decoding_relu

"""
Weights Initialization
"""


def he_init(model):
    if isinstance(model, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(model.weight, mode="fan_out", nonlinearity="relu")

    if isinstance(model, nn.Linear):
        size = model.weight.size()
        fan_out = size[0]
        fan_in  = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        model.weight.data.normal_(0.0, variance)

    elif isinstance(model, GroupNorm):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)


class AffineTransformation(nn.Module):
    def __init__(self):
        super(AffineTransformation).__init__()
        in_ch     = args_at.in_ch
        in_nf     = args_at.in_fs
        filters   = [in_nf, in_nf * 2, in_nf * 4, in_nf * 8, in_nf * 16]

        self.enc1 = encoding_relu(in_ch, filters[0], 3, 2, 3, 8)
        self.enc2 = encoding_relu(filters[0], filters[1], 3, 2, 3, 8)
        self.enc3 = encoding_relu(filters[1], filters[2], 3, 2, 3, 8)
        self.enc4 = encoding_relu(filters[2], filters[3], 3, 2, 3, 8)
        self.enc5 = encoding_relu(filters[3], filters[4], 3, 2, 3, 8)

        self.avp1 = nn.AvgPool3d(3, 2)

