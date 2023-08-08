import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    

def correlation_coefficient_loss(fixed, warped):
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
    

def variatinal_energy_loss(flow):
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


def orthogonal_loss(matrix):
        """
        This loss represents the orthogonal loss of PAM
        matrix has shape (batch, 3, 3)
        """
        # Compute the orthogonal loss
        identity        = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
        matrix          = identity[None, ...] + matrix[:, :3, :3]

        covar           = torch.matmul(matrix, matrix.transpose(1, 2)) + 1e-5
        eigenvals, _    = torch.linalg.eigvals(covar, eigenvectors=True)

        eigenvals       = torch.square(eigenvals)
        loss            = (eigenvals + 1e-5) + 1.0 / (eigenvals + 1e-5)
        loss            = - 6. + torch.sum(loss)
        return loss
     

def dice_loss(target, pred, smooth=1e-5):
    num_classes = pred.size(1)
    dice = 0
    class_count = 0
    
    for class_idx in range(num_classes):
        class_pred = pred[:, class_idx, ...]
        class_target = (target == class_idx).float()
        
        intersection = torch.sum(class_pred * class_target)
        cardinality = torch.sum(class_pred) + torch.sum(class_target)
        
        # Check if the class is present in the target
        if torch.sum(class_target) > 0:
            dice_class = (2.0 * intersection + smooth) / (cardinality + smooth)
            dice += dice_class
            class_count += 1
    
    # Check if any class was present in the target
    if class_count > 0:
        dice_loss = 1 - dice / class_count
    else:
        dice_loss = torch.tensor(0.0)
    
    return dice_loss


def xent_segmentation(target, pred):
    # Flatten the predictions and targets
    pred_flat = pred.view(-1, pred.size(1))
    target_flat = target.view(-1, pred.size)

    # Calculate cross entropy loss
    loss = F.cross_entropy(pred_flat, target_flat)

    return loss
