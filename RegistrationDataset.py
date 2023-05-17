import torch
from   torch.utils import data
import numpy as np

from   libs.frida.io         import ImageLoader, ReadVolume
from   libs.frida.transforms import  ZeroOneScaling, ToNumpyArray

from SimpleITK import ReadImage, GetArrayFromImage


def zero_pad_inplace(arr):
    arr[:,  :,  0]  = 0
    arr[:,  :, -1]  = 0
    arr[:,  0,  :]  = 0
    arr[:, -1,  :]  = 0
    arr[0,  :,  :]  = 0
    arr[-1,  :,  :] = 0
    return arr


def load_image(path):
    im = ReadImage(path)
    im = GetArrayFromImage(im)
    im = (im - im.min()) / (im.max() - im.min())
    im = im[-160:] # thorax # this needs to be checked 
    im = zero_pad_inplace(im)
    return im


def np2torch(arr):
    arr = arr.transpose(1, 2, 0)
    arr = torch.from_numpy(arr).type(torch.float32)
    arr = arr[None, :]
    return arr


class RegistrationDataSet(data.Dataset):
    def __init__(self,
                 path_dataset: str,
                 input_shape : tuple,
                 transform   = None
                 ):
        
        self.dataset     = path_dataset
        self.input_shape = input_shape
        self.indices     = path_dataset.index.values.copy()
        self.transform   = transform
        self.random_seed = int(0)
        self.inp_dtype   = torch.float32
        self.log         = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,
                    index: int):

        # Select the sample
        image_path = self.dataset.iloc[index]
        image_path = str(image_path.squeeze().dicom_path)
        fx = load_image(image_path)

        # Make random pair
        moving_path = self.dataset.sample(n=1)
        moving_path = str(moving_path.squeeze().dicom_path)
        mv = load_image(moving_path)

        # to torch
        fx = np2torch(fx)
        mv = np2torch(mv)

        return fx, mv