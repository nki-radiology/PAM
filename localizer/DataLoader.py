import os

import numpy as np
import torch
from skimage.io import imread
from torch.utils import data
from skimage.transform import resize
from torchvision import transforms

from pathlib import Path


def resize_img(img, target_size=256):
    img = resize(image=img, output_shape=(target_size, target_size), preserve_range=True)
    img = img / 420
    img_aux = np.dstack((img, img, img))
    return img_aux


def read_npy(filename):
    img       = np.load(filename)
    x1, x2, y = np.rollaxis(img, 2, 0)
    return x1, x2, y


class CustomDataSet(data.Dataset):
    def __init__(self,
                 path_dataset   : str,
                 img_size  : int,
                 transform = None
                 ):
        self.filenames = path_dataset
        self.img_size  = img_size
        self.transform = transform
        self.inp_dtype = torch.float32

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        # Select the sample
        input_dcm = self.filenames[index]

        # Load input and target
        x_1, x_2, y = read_npy(input_dcm)
        y           = torch.from_numpy(y[0][0])

        # Resize and moving axes
        x_1 = resize_img(x_1, self.img_size)
        x_1 = np.moveaxis(x_1, [2], [0])
        x_2 = resize_img(x_2, self.img_size)
        x_2 = np.moveaxis(x_2, [2], [0])

        # Transformation
        if self.transform is not None:
            x_1, x_2 = self.transform(x_1, x_2)

        # Typecasting
        x_1 = torch.from_numpy(x_1).type(self.inp_dtype)
        x_2 = torch.from_numpy(x_2).type(self.inp_dtype)

        return x_1, x_2, y


# Verifying if the dataloader is working properly
"""
inputs              = '../../../../../DATA/laura/tcia_temp/train/'
training_dataset    = CustomDataSet(path_dataset = inputs,
                                 img_size     = 256,
                                 transform    = None)
training_dataloader = data.DataLoader(dataset    = training_dataset,
                                      batch_size = 10,
                                      shuffle    = True)

x, y, z   = next(iter(training_dataloader))
print(f'x = shape: {x.shape}, class: {x.unique()}, type: {x.dtype}')
print(f'x = min:   {x.min()}, max:   {x.max()}')
print(f'y = shape: {y.shape}, class: {y.unique()}, type: {y.dtype}')
print(f'y = min:   {y.min()}, max:   {y.max()}')
print(f'z = shape: {z.shape}, class: {z.unique()}, type: {z.dtype}')
"""