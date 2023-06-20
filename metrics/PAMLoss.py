import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    

def correlation_coefficient_loss(self, fixed, warped):
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
    

def variatinal_energy_loss(self, flows):
        """
        This loss represents the total variation loss or the elastic loss of PAM
        flow has shape (batch, 3, D1, D2, D3)
        """
        flow = flow[None, :]
        dy   = flow[:, :, 1:,  :,  :] - flow[:, :, :-1, :  , :  ]
        dx   = flow[:, :,  :, 1:,  :] - flow[:, :, :  , :-1, :  ]
        dz   = flow[:, :,  :,  :, 1:] - flow[:, :, :  , :  , :-1]
        d    = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

        return d / 3.0


def dice_loss(true, logits, eps=1e-7):
    """
    Computes the Sørensen–Dice loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen–Dice loss.

    Source:
        https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """

    num_classes = logits.shape[1]

    if num_classes == 1:
        # Convert true to one-hot encoded tensor
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)

        # Compute positive and negative probabilities
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)

    else:
        # Convert true to one-hot encoded tensor
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

        # Compute class probabilities using softmax
        probas = F.softmax(logits, dim=1)

    true_1_hot = true_1_hot.type(logits.type())

    # Calculate intersection and cardinality
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)

    # Compute dice loss
    dice_loss = (2. * intersection / (cardinality + eps)).mean()

    # Return negated dice loss
    return (1 - dice_loss)
