import os
import torch
import argparse
import torch.nn as     nn
from   pathlib  import Path


def str2bool(v):
    if   v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True    
    elif v.lower() in ( 'no', 'false', 'f', 'n', '0'):
        return False
    else: 
        return argparse.ArgumentTypeError('Boolean value expected!')


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Directory created!")
    else:
        print("Directory already created")


def cuda_seeds():
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark    = False


def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
            
def read_train_data(path_input):
    path      = Path(path_input)
    filenames = list(path.glob('*.jpeg'))
    return filenames