import torch


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
    dy   = flow[:, :, 1:,  :,  :] - flow[:, :, :-1, :  , :  ]
    dx   = flow[:, :,  :, 1:,  :] - flow[:, :, :  , :-1, :  ]
    dz   = flow[:, :,  :,  :, 1:] - flow[:, :, :  , :  , :-1]
    d    = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

    return d / 3.0


def total_elastic_loss(fixed, moving, flows):
    """
    Deformation network loss -> To test in the future
    :param fixed : fixed image
    :param moving: moving image
    :param flows : flows
    :return      : correlation coefficient loss plus total variation loss
    """
    sim_loss = pearson_correlation(fixed, moving)

    # Regularize all flows
    if len(fixed.size() ) == 4:  # (N, C, H, W)
        reg_loss = elastic_loss_2D(flows)
    else:
        reg_loss = sum([elastic_loss_3D(flow) for flow in flows])

    return sim_loss + reg_loss


def elem_sym_polys_of_eigen_values(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    sigma1 = M[0][0] + M[1][1] + M[2][2]

    sigma2 = (M[0][0] * M[1][1] + M[1][1] * M[2][2] + M[2][2] * M[0][0]) - \
             (M[0][1] * M[1][0] + M[1][2] * M[2][1] + M[2][0] * M[0][2])

    sigma3 = (M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1]) - \
             (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])

    return sigma1, sigma2, sigma3


def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return (M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1]) - \
           (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])


def affine_loss(matrix_A):
    """
    This loss represents the orthogonality loss and determinant loss of PAM
    flow has shape (batch, 4, 3)
    adapted from: https://github.com/microsoft/Recursive-Cascaded-Networks/blob/0709490fe010e4acc6357990d1d871c170e3ef31/network/base_networks.py#L193
    """
    # Determinant loss: it should be close to one
    # det      = det3x3(matrix_A)
    # det      = det - 1
    # det_loss = (det * det)/2

    # --------------------------------- Using just pytorch ---------------------------------
    det      = torch.linalg.det(matrix_A) - 1.0
    det_loss = (det * det)/2
    # --------------------------------------------------------------------------------------

    # Orthogonality loss: A'A eigenvalues should be close to 1
    I          = torch.eye(3, dtype=torch.float32, device='cuda')
    eps        = 1e-5
    epsI       = I * eps
    #A_t        = matrix_A.transpose(-2, -1)
    #covariance = torch.matmul(A_t, matrix_A) + epsI
    #s1, s2, s3 = elem_sym_polys_of_eigen_values(covariance)
    #ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
    #ortho_loss = torch.sum(ortho_loss)

    # --------------------------------- Using just pytorch ---------------------------------
    covariance = torch.matmul(matrix_A, matrix_A.T.conj()) + epsI
    eigvalsh = torch.linalg.eigvalsh(covariance)
    ortho_loss = (eigvalsh + 1e-5) + ((1 + 1e-5) ** 2) / (eigvalsh + 1e-5)
    ortho_loss = -6 + torch.sum(ortho_loss)
    # --------------------------------------------------------------------------------------

    return det_loss, ortho_loss
