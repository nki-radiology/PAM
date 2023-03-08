import os
import wandb
import torch
import argparse
import pandas             as pd
import torch.nn           as nn
import matplotlib.pyplot  as plt
from   pathlib        import Path


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
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
            
def read_train_data(path_input):
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


def save_images_weights_and_biases(table_name, path_to_save, fixed_img, moving_img, w0_img, w1_img):
    table = wandb.Table(columns=['Fixed Image', 'Moving Image', 'Affine Reg. Image', 'Deformation Reg. Image'], allow_mixed_types = True)
    
    saving_examples_folder = path_to_save
    
    #PIL VERSION
    transform = T.ToPILImage()    
    fixed_img = transform(fixed_img[:,:,:,:,50].squeeze())
    moving_img = transform(moving_img[:,:,:,:,50].squeeze())
    affine_img = transform(w0_img[:,:,:,:,50].squeeze())
    deformation_img = transform(w1_img[:,:,:,:,50].squeeze())

    fixed_img.show()                              
    fixed_img.save(saving_examples_folder + "fixed_image.jpg")    
    moving_img.show() 
    moving_img.save(saving_examples_folder + "moving_image.jpg")    
    affine_img.show() 
    affine_img.save(saving_examples_folder + "affine_image.jpg")    
    deformation_img.show()
    deformation_img.save(saving_examples_folder + "deformation_image.jpg")    
    
    table.add_data(
        wandb.Image(Image.open(saving_examples_folder + "fixed_image.jpg")),
        wandb.Image(Image.open(saving_examples_folder + "moving_image.jpg")),
        wandb.Image(Image.open(saving_examples_folder + "affine_image.jpg")),
        wandb.Image(Image.open(saving_examples_folder + "deformation_image.jpg")),
    )
    
    wandb.log({table_name: table})