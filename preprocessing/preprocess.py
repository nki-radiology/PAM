# example
# python preprocess.py --body-part thorax --input /data/groups/beets-tan/s.trebeschi/MPM_FOLLOWUP/0.images --output /data/groups/beets-tan/s.trebeschi/MPM_FOLLOWUP/1.processed/

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--body-part',          type = str, default = 'thorax',         help = 'body part to train on')
parser.add_argument('--input',              type = str, default = None,             help = 'folder that contains the dataset')
parser.add_argument('--output',             type = str, default = None,             help = 'folder that contains the dataset')

parser.add_argument('--debug' ,             type = bool, default = False,           help = 'debug mode')

PARAMS                                      = parser.parse_args()

IMG_DIM             = [192, 192, 192]        
BODY_PART           = PARAMS.body_part
INPUT               = PARAMS.input
OUTPUT              = PARAMS.output
DEBUG               = PARAMS.debug

if DEBUG:
    import pdb; pdb.set_trace()

print('#############################################')
print('Parameters:')
print(PARAMS)
print('#############################################')

# scan folder
import os
import pandas   as pd
import pydicom

from pathlib                        import Path

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
    if os.path.isdir(INPUT):
        path        = Path(INPUT)
        candidates  = list(path.glob('*.nrrd'))
        candidates += list(path.glob('*.nii.gz'))
        candidates += list_dicom_folders()
        candidates  = [str(c) for c in candidates]
        dataset     = pd.DataFrame(candidates, columns=['images'])
    elif os.path.isfile(INPUT) and INPUT.endswith('.csv'):
        dataset     = pd.read_csv(INPUT)
    else:
        raise ValueError('Invalid input')

    return dataset


# define imaging transform functions
import SimpleITK as sitk
from SimpleITK                  import CastImageFilter
from SimpleITK                  import ClampImageFilter

clamp = ClampImageFilter()
clamp.SetUpperBound(300)
clamp.SetLowerBound(-120)

cast = CastImageFilter()
cast.SetOutputPixelType(sitk.sitkInt16)


# define transforms
from frida.io                   import ImageLoader
from frida.io                   import ReadVolume

from frida.transforms           import TransformFromITKFilter
from frida.transforms           import Resample
from frida.transforms           import PadAndCropTo

from Localizer                  import CropThorax

loader = ImageLoader(
    ReadVolume(),
    CropThorax(tolerance=25),
    Resample(2),
    PadAndCropTo((192, 192, 192), cval=-1000),
    TransformFromITKFilter(clamp),
    TransformFromITKFilter(cast)
)

# iterate over all images
print ('Loading images...')
dataset = data_inventory()
print ('Loaded {} images'.format(len(dataset)))

log = []
for i, row in dataset.iterrows():
    print ('Loading image {} of {}'.format(i+1, len(dataset)))
    try:
        image = loader(row['images'])
    except:
        print ('Error loading image')
        continue

    print ('Saving image...')
    filename = str(i).zfill(12)
    filename = os.path.join(OUTPUT, filename + ".nii.gz")
    sitk.WriteImage(image, row['images'])

    log.append({
        'input': row['images'],
        'output': filename
    })

    print ('Done')

log = pd.DataFrame(log)
log.to_csv(os.path.join(OUTPUT, 'log.csv'), index=False)