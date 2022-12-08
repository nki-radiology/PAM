import torch
import torch.nn.functional as F

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon    = F.relu(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats

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


