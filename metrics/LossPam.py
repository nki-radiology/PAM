import math
import torch
import numpy as np
import torch.nn.functional as F


def voxel_morph_loss(self, y_true, y_pred):

    Ii = y_true
    Ji = y_pred

    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims if self.win is None else self.win  # smooth

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-6)
    return 1 - torch.mean(cc)


# Loss including smoothing factor
def avp_loss(self, y_true, y_pred, smooth=9):
    l_true = torch.nn.AvgPool3d(kernel_size=smooth, stride=1, padding=0)  # 1
    l_pred = torch.nn.AvgPool3d(kernel_size=smooth, stride=1, padding=0)  # 1
    y_true = l_true(y_true)
    y_pred = l_pred(y_pred)

    # Flatten
    flatten_fixed = torch.flatten(y_true, start_dim=1)
    flatten_warped = torch.flatten(y_pred, start_dim=1)

    # Compute the mean
    mean_fixed = torch.mean(flatten_fixed)
    mean_warped = torch.mean(flatten_warped)

    # Compute the variance
    var_fixed = torch.mean((flatten_fixed - mean_fixed) ** 2)
    var_warped = torch.mean((flatten_warped - mean_warped) ** 2)

    # Compute the covariance
    cov_fix_war = torch.mean((flatten_fixed - mean_fixed) * (flatten_warped - mean_warped))
    eps = 1e-6

    # Compute the correlation coefficient loss
    pearson_r = cov_fix_war / torch.sqrt((var_fixed + eps) * (var_warped + eps))
    raw_loss = 1 - pearson_r

    return raw_loss


def pearson_correlation(fixed, warped):
    """
    This loss represents the correlation coefficient loss of PAM
    fixed and warped have shapes (batch, 1, 192, 192, 160)
    """

    # Flatten
    flatten_fixed  = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    # Compute the mean
    mean_fixed     = torch.mean(flatten_fixed)
    mean_warped    = torch.mean(flatten_warped)

    # Compute the variance
    var_fixed      = torch.mean((flatten_fixed - mean_fixed) ** 2)
    var_warped     = torch.mean((flatten_warped - mean_warped) ** 2)

    # Compute the covariance
    cov_fix_war    = torch.mean((flatten_fixed - mean_fixed) * (flatten_warped - mean_warped))
    eps            = 1e-6

    # Compute the correlation coefficient loss
    pearson_r      = cov_fix_war / torch.sqrt((var_fixed + eps) * (var_warped + eps))
    raw_loss       = 1 - pearson_r

    return raw_loss



def elastic_loss_2D(flow):
    """
    flow has shape (batch, 2, 192, 192)
    Loss for 2D dataset
    """
    dx = (flow[..., 1:, :] - flow[..., :-1, :]) ** 2
    dy = (flow[..., 1:   ] - flow[..., :-1   ]) ** 2

    d = torch.mean(dx) + torch.mean(dy)

    return d / 2.0



def elastic_loss_3D(flow):
    """
    This loss represents the total variation loss or the elastic loss of PAM
    flow has shape (batch, 3, 192, 192, 160)
    """
    flow = flow[None, :]
    dy   = flow[:, :, 1:,  :,  :] - flow[:, :, :-1, :  , :  ]
    dx   = flow[:, :,  :, 1:,  :] - flow[:, :, :  , :-1, :  ]
    dz   = flow[:, :,  :,  :, 1:] - flow[:, :, :  , :  , :-1]
    d    = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

    return d / 3.0


def total_loss(fixed, moving, flows):
    """
    Deformation network loss
    :param fixed : fixed image
    :param moving: moving image
    :param flows : flows
    :return      : correlation coefficient loss plus total variation loss
    """
    sim_loss = pearson_correlation(fixed, moving)

    # Regularize all flows
    if len(fixed.size()) == 4:  # (N, C, H, W)
        reg_loss = elastic_loss_2D(flows)
    else:
        reg_loss = sum([elastic_loss_3D(flow) for flow in flows])

    return sim_loss, reg_loss

