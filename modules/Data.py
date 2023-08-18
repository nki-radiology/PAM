import os
import numpy    as np
import pandas   as pd
import torch 

from torch.utils                    import data
from SimpleITK                      import ReadImage
from SimpleITK                      import GetArrayFromImage
from SimpleITK                      import GetImageFromArray
from SimpleITK                      import WriteImage
from pydicom                        import dcmread
from pathlib                        import Path

from config                         import PARAMS

IMG_DIM             = PARAMS.img_dim
BODY_PART           = PARAMS.body_part
DATASET_FOLDER      = PARAMS.dataset_folder
DATASET_FOLLOWUP    = PARAMS.dataset_followup


def data_inventory():
    path        = Path(DATASET_FOLDER)
    candidates  = list(path.glob('*.nrrd'))
    candidates += list(path.glob('*.nii.gz'))
    dataset     = pd.DataFrame(candidates, columns=['images'])

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


def load_dicom_tagssafely(path, prefix = ''):
        # wraps metatags loading around a try-catch
        # attach a prefix to the fields if needed
        result = {}

        try:
            dcm = os.path.join(path, os.listdir(path)[0])
            ds  = dcmread(dcm)

            tags = (
                0x00080020, # Study Date
                0x00081030, # Study Description
                0x00180060, # KVP
                0x00280030, # Pixel Spacing
                0x00180050, # Slice Thickness
                0x00180088, # Spacing Between Slices
                0x00189306, # Single Collimation Width
                0x00189307, # Total Collimation Width
                0x00181151, # X-Ray Tube Current
                0x00181210, # Convolution Kernel
                0x00181150, # Exposure Time
                0x00189311  # Spiral Pitch Factor
            )

            result = dict()
            for t in tags:
                try:
                    descr = ds[t].description()
                    descr = descr.replace(' ', '').replace('-', '')
                    descr = prefix + descr.lower()
                    result.update({descr: ds[t].value})
                except:
                    pass
        except:
            print(' - [failed] while loading of the DICOM tags. ' )

        return result


def save(tensor, path):
    if path.endswith('.nii.gz') or path.endswith('.nrrd'):
        # move the channel dimension to the end
        tensor = tensor.permute(0, 2, 3, 4, 1)
        # convert to numpy array
        tensor = tensor.detach().cpu().numpy().squeeze()
        tensor = tensor.astype(np.float32)
        # convert to SimpleITK image
        tensor = GetImageFromArray(tensor)
        # save image
        WriteImage(tensor, path)

    elif path.endswith('.npy'):
        # convert to numpy array
        tensor = tensor.detach().cpu().numpy().squeeze()
        tensor = tensor.astype(np.float32)
        # save image
        np.save(path, tensor)
        
    else:
        raise ValueError('extension not recognized')



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
    def __init__(self, mode='train'):
        self.dataset_fixed     = SimpleDataset()
        self.dataset_moving    = SimpleDataset()
        self.mode              = mode
    
    def __len__(self):
        return len(self.dataset_fixed)

    def __getitem__(self, idx):
        fixed       = self.dataset_fixed.sample()
        moving      = self.dataset_moving.sample()
        return fixed, moving


class FollowUpDataset(data.Dataset):
    def __init__(self):
        self.dataset = pd.read_csv(DATASET_FOLLOWUP)

        if 'baseline' not in self.dataset.columns:
            raise ValueError('no baseline column found in the follow-up dataset')
        elif 'followup' not in self.dataset.columns:
            raise ValueError('no followup column found in the follow-up dataset')
        else:
            print(' - [info] follow-up dataset loaded successfully')
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        baseline_path       = self.dataset.iloc[idx]['baseline']
        baseline_image      = load(baseline_path)

        followup_path       = self.dataset.iloc[idx]['followup']
        followup_image      = load(followup_path)

        return baseline_image, followup_image



