# example
# python preprocess.py --body-part thorax --input /data/groups/beets-tan/s.trebeschi/MPM_FOLLOWUP/0.images --output /data/groups/beets-tan/s.trebeschi/MPM_FOLLOWUP/1.processed/

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--body-part',          type = str, default = 'thorax',         help = 'body part to train on')
parser.add_argument('--input',              type = str, default = None,             help = 'folder that contains the dataset')
parser.add_argument('--input-masks',        type = str, default = None,             help = 'folder that contains the dataset')
parser.add_argument('--output',             type = str, default = None,             help = 'folder that contains the dataset')
parser.add_argument('--output-masks',       type = str, default = None,             help = 'folder that contains the dataset')

parser.add_argument('--debug' ,             type = bool, default = False,           help = 'debug mode')

PARAMS                                      = parser.parse_args()

IMG_DIM             = [192, 192, 192]        
BODY_PART           = PARAMS.body_part

INPUT               = PARAMS.input
INPUT_MASKS         = PARAMS.input_masks

OUTPUT              = PARAMS.output
OUTPUT_MASKS        = PARAMS.output_masks

DEBUG               = PARAMS.debug

if DEBUG:
    import pdb; pdb.set_trace()


print('#############################################')
print('Parameters:')
print(PARAMS)
print('#############################################')

import os
import pandas   as pd
import pydicom

from pathlib                        import Path

import SimpleITK as sitk
from SimpleITK                  import CastImageFilter
from SimpleITK                  import ClampImageFilter

from frida.io                   import ImageLoader
from frida.io                   import ReadVolume

from frida.transforms           import TransformFromITKFilter
from frida.transforms           import Resample
from frida.transforms           import PadAndCropTo

from Localizer                  import CropThorax
from Localizer                  import CropAdbomen
from Localizer                  import LinkedSmartCrop


def list_dicom_folders():
    dicom_dirs = []

    for dirpath, dirnames, filenames in os.walk(INPUT):
        for file in filenames:
            filepath = os.path.join(dirpath, file)
            try:
                pydicom.dcmread(filepath)
                dicom_dirs.append(dirpath)
                break  
            except:
                pass

    return dicom_dirs


def data_inventory():
    print ('Loading images...')

    # load images
    if os.path.isdir(INPUT):
        path        = Path(INPUT)
        candidates  = list(path.glob('*.nrrd'))
        candidates += list(path.glob('*.nii.gz'))
        candidates += list_dicom_folders()
        candidates  = [str(c) for c in candidates]
        dataset     = pd.DataFrame(candidates, columns=['image'])
    elif os.path.isfile(INPUT) and INPUT.endswith('.csv'):
        dataset     = pd.read_csv(INPUT)
    else:
        raise ValueError('Invalid input')
    
    # load masks
    if INPUT_MASKS is not None:
        if INPUT_MASKS.endswith('.csv'):
            dataset_masks = pd.read_csv(INPUT_MASKS)
            dataset.merge(dataset_masks, on='image', how='left')
        else:
            ValueError('Needs a list of segmentation masks')
        print(' -- no masks provided, skipping')

    print (' -- Loaded {} images'.format(len(dataset)))

    return dataset


def init_loaders():
    # +300HU cortical bone
    # -120HU fat
    clamp = ClampImageFilter()
    clamp.SetUpperBound(300)
    clamp.SetLowerBound(-120)

    # cast to int16 to save space on disk
    cast = CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkInt16)

    # crop objects
    if BODY_PART == 'thorax':
        crop_obj = CropThorax(tolerance=25)
    elif BODY_PART == 'abdomen':
        crop_obj = CropAdbomen(tolerance=25)
    else:
        raise ValueError('Invalid body part')

    # loader objects
    loader = ImageLoader(
        ReadVolume(),
        crop_obj,
        Resample(2),
        PadAndCropTo((192, 192, 192), cval=-1000),
        TransformFromITKFilter(clamp),
        TransformFromITKFilter(cast)
    )

    mask_loader = ImageLoader(
        ReadVolume(),
        LinkedSmartCrop(crop_obj),
        Resample(2, sitk.sitkNearestNeighbor),
        PadAndCropTo((192, 192, 192), cval=0)
    )

    return loader, mask_loader


def preprocess(dataset):
    loader, mask_loader = init_loaders()

    log = []
    for i, row in dataset.iterrows():
        print ('Loading image {} of {}'.format(i+1, len(dataset)))

        entry = {
            'input_image': row['image'],
            'input_mask': row['mask'] if INPUT_MASKS is not None else None,
        }

        try:
            image = loader(row['image'])
            filename = str(i).zfill(12)
            filename = os.path.join(OUTPUT, filename + ".nii.gz")
            sitk.WriteImage(image, filename)
            entry['output_image'] = row['image']

        except:
            print (' -- Error loading image')
            continue

        if INPUT_MASKS is not None:
            try:
                mask = mask_loader(row['mask'])
                filename = str(i).zfill(12)
                filename = os.path.join(OUTPUT_MASKS, filename + ".nii.gz")
                sitk.WriteImage(mask, filename)
                entry['output_mask'] = row['mask']

            except:
                print (' -- Error loading mask, skipping')

        log.append(entry)
        print (' -- Done')
    
    return pd.DataFrame(log)


if __name__ == "__main__":

    dataset = data_inventory()
    log = preprocess(dataset)
    
    log.to_csv(os.path.join(OUTPUT, 'log.csv'), index=False)
