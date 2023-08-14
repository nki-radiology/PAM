
import numpy as np
import pandas as pd

import torch 
from torch.utils                    import data

from SimpleITK                      import ReadImage
from SimpleITK                      import GetArrayFromImage

from pathlib                        import Path

from config                         import PARAMS


IMG_DIM             = PARAMS.img_dim
BODY_PART           = PARAMS.body_part
DATASET_CSV         = PARAMS.dataset_csv
DATASET_FOLDER      = PARAMS.dataset_folder


def data_inventory():
    if DATASET_CSV is None:
        path        = Path(DATASET_FOLDER)
        filenames   = list(path.glob('*.nrrd'))
        filenames  += list(path.glob('*.nii.gz'))
        dataset     = pd.DataFrame(filenames, columns=['images'])
    else:
        dataset     = pd.read_csv(DATASET_CSV)

    return dataset


def load(path):
    # load image
    image = ReadImage(path)
    image = GetArrayFromImage(image)

    # normalize
    image = (image - image.min()) / (image.max() - image.min())

    # crop to body part, thorax by default is the upper part
    z = IMG_DIM[-1]
    if BODY_PART == 'thorax':
        image = image[-z:] 
    elif BODY_PART == 'abdomen':
        image = image[:z] 
    else:
        raise ValueError('body part not recognized')
    
    # zero padding in place
    image[:,  :,  0]   =   image[:,  :, -1]   = 0
    image[:,  0,  :]   =   image[:,  -1, :]   = 0
    image[0,  :,  :]   =   image[-1,  :, :]   = 0

    # convert to torch tensor
    image = torch.from_numpy(image).type(torch.float32)
    image = image[None, :]

    return image


class SimpleDataset(data.Dataset):
    def __init__(self):
        self.dataset = data_inventory()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset.iloc[idx]['images']
        image = load(image_path)
        return image
    
    def sample(self):
        idx = np.random.randint(0, len(self.dataset))
        image = self[idx]
        return image


class RegistrationDataset(data.Dataset):
    def __init__(self):
        self.dataset_fixed      = SimpleDataset()
        self.dataset_moving     = SimpleDataset()
    
    def __len__(self):
        return len(self.dataset_fixed)

    def __getitem__(self, idx):
        fixed   = self.dataset_fixed.sample()
        moving  = self.dataset_moving.sample()
        return fixed, moving




