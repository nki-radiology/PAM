import torch
from   torch.utils import data
import numpy as np

from SimpleITK import ReadImage, GetArrayFromImage

from config import PARAMS


TOTSEG_LABELS_THORAX = [
	7, 8, 											# aorta and vena cava
	13, 14, 15, 16, 17, 							# lung
	23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, # vertebrae
	42, 43, 										# esophagus and trachea
	44, 45, 46, 47, 48, 49, 						# heart and pulm. artery
	58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, # ribs left
	70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, # ribs right
	84, 85, 										# scapelas
	86, 87											# claviculas
]

TOTSEG_LABELS_ABDOMEN = [
	1, 2, 3, 4, 5, 6,					            # large upper organs
	7, 8, 9, 							            # aorta and venas
	10, 11, 12, 						            # pancreas and adr glands
	18, 19, 20, 21, 22, 				            # vertebrae
	51, 52, 53, 54, 					            # iliac vena and artery
	55, 56, 57, 						            # bowels
	90, 91, 92, 						            # hips and sacrum
	94, 95, 96, 97, 98, 99, 102, 103, 	            # gluteous and iliopsoas
	104									            # bladder
]


def zero_pad_inplace(arr):
    arr[:,  :,  0]  = 0
    arr[:,  :, -1]  = 0
    arr[:,  0,  :]  = 0
    arr[:, -1,  :]  = 0
    arr[0,  :,  :]  = 0
    arr[-1,  :,  :] = 0
    return arr


def load_image(path, body_part):
    im = ReadImage(path)
    im = GetArrayFromImage(im)
    im = (im - im.min()) / (im.max() - im.min())
    
    z = PARAMS.img_dim[-1]
    if body_part == 'thorax':
        im = im[-z:] 
    elif body_part == 'abdomen':
        im = im[:z] 
    else:
        raise ValueError('body part not recognized')
    im = zero_pad_inplace(im)

    return im


def get_num_classes(body_part):
    if body_part == 'thorax':
        return len(TOTSEG_LABELS_THORAX)
    elif body_part == 'abdomen':
        return len(TOTSEG_LABELS_ABDOMEN)
    else:
        raise ValueError('body part not recognized')


def filter_segmentation_mask(segmentation_mask, labels):
    filtered_mask = np.zeros_like(segmentation_mask)
    for i, label in enumerate(labels):
        filtered_mask[segmentation_mask == label] = i + 1

    return filtered_mask


def load_segmentation(path, body_part):
    seg = ReadImage(path)
    seg = GetArrayFromImage(seg)

    z = PARAMS.img_dim[-1]
    if body_part == 'thorax':
        seg = seg[-z:] 
        seg = filter_segmentation_mask(seg, TOTSEG_LABELS_THORAX)
    elif body_part == 'abdomen':
        seg = seg[:z] 
        seg = filter_segmentation_mask(seg, TOTSEG_LABELS_ABDOMEN)
    else:
        raise ValueError('body part not recognized')
    
    seg = zero_pad_inplace(seg)
    return seg


def np2torch(arr):
    #arr = arr.transpose(1, 2, 0)
    arr = torch.from_numpy(arr).type(torch.float32)
    arr = arr[None, :]
    return arr


class PAMDataset(data.Dataset):
    def __init__(self,
                 dataset,
                 input_shape : tuple,
                 transform   = None,
                 body_part   = 'thorax'
                 ):
        
        self.dataset     = dataset
        self.input_shape = input_shape
        self.indices     = dataset.index.values.copy()
        self.transform   = transform
        self.random_seed = int(0)
        self.inp_dtype   = torch.float32
        self.log         = []
        self.body_part   = body_part

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,
                    index: int):

        def load_from_dataframe_entry(entry):
            image_path = entry.squeeze().dicom_path
            image      = load_image(str(image_path), self.body_part)
            result     = np2torch(image)

            if 'dicom_path_seg' in self.dataset.columns:
                seg_path = str(entry.squeeze().dicom_path_seg)
                seg      = load_segmentation(seg_path, self.body_part)
                seg      = np2torch(seg)
                result   = (result, seg)

            return result
        
        fx = load_from_dataframe_entry(self.dataset.iloc[index])
        mv = load_from_dataframe_entry(self.dataset.sample(n=1))

        return fx, mv