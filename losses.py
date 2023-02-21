import torch
import torch.nn as nn
import torch.nn.functional as F

    

def kl_divergence(mu, log_var):
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    return kl_loss

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


class Cross_Correlation_Loss(nn.Module):

    def __init__(self):
        super(Cross_Correlation_Loss, self).__init__()

    def pearson_correlation(self, fixed, warped):
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



class Energy_Loss(nn.Module):

    def __init__(self):
        super(Energy_Loss, self).__init__()

    def elastic_loss_2D(self, flow):
        """
        flow has shape (batch, 2, 192, 192)
        Loss for 2D dataset
        """
        dx = (flow[..., 1:, :] - flow[..., :-1, :]) ** 2
        dy = (flow[..., 1:   ] - flow[..., :-1   ]) ** 2

        d = torch.mean(dx) + torch.mean(dy)

        return d / 2.0



    def elastic_loss_3D(self, flow):
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


    def energy_loss(self, flows):

        if len(flows.size()) == 4:  # (N, C, H, W)
            reg_loss = self.elastic_loss_2D(flows)
        else:
            reg_loss = sum([self.elastic_loss_3D(flow) for flow in flows])
        
        return reg_loss


class Total_Loss(nn.Module):

    def __init__(self):
        super(Total_Loss, self).__init__()
        self.pearson_correlation = Cross_Correlation_Loss()
        self.penalty_deformation = Energy_Loss()


    def total_loss(self, fixed, moving, flows):
        """
        Deformation network loss
        :param fixed : fixed image
        :param moving: moving image
        :param flows : flows
        :return      : correlation coefficient loss plus total variation loss
        """
        sim_loss = self.pearson_correlation.pearson_correlation(fixed, moving)

        # Regularize all flows
        energy_loss = self.penalty_deformation.energy_loss(flows)

        return sim_loss, energy_loss