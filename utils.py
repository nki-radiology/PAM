import os
import wandb
import torch
import argparse
import pandas             as pd
import torch.nn           as nn
from   pathlib        import Path
from   PIL                          import Image, ImageOps
import torchvision.transforms       as     T


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
        print("Directory created in: ", path)
    else:
        print("Directory already created: ", path)


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
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
            
def read_2D_train_data(path_input):
    path      = Path(path_input)
    filenames = list(path.glob('*.jpeg'))
    data_index= []

    i = 0
    for f in filenames:
        data_index.append(i)
        i += 1

    train_data = list(zip(data_index, filenames))
    train_data = pd.DataFrame(train_data, columns=['tcia_idx', 'dicom_path'])
    return train_data


def read_3D_train_data(path_input):
    path      = Path(path_input)
    filenames = list(path.glob('*.nrrd'))
    data_index= []

    for f in filenames:
        data_index.append(int(str(f).split('/')[7].split('_')[0])) # Number 8 can vary according to the path of the images

    train_data = list(zip(data_index, filenames))
    train_data = pd.DataFrame(train_data, columns=['tcia_idx', 'dicom_path'])
    return train_data


    
def read_3D_survival_train_valid_data(filename):
    data = pd.read_csv(filename)
    print('total len: ', len(data))
    train_data = data.loc[data['fold'] == 'train']
    train_data = train_data[['PRIOR_PATH_NRRD', 'SUBSQ_PATH_NRRD', 'Y1Survival']]
    print('only training len: ', len(train_data))
    valid_data = data.loc[data['fold'] == 'valid']
    valid_data = valid_data[['PRIOR_PATH_NRRD', 'SUBSQ_PATH_NRRD', 'Y1Survival']]
    print('only validation len: ', len(valid_data))
    return train_data, valid_data
