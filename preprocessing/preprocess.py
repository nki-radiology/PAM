from config         import PARAMS

IMG_DIM             = PARAMS.img_dim
BODY_PART           = PARAMS.body_part
DATASET_CSV         = PARAMS.dataset_csv
DATASET_FOLDER      = PARAMS.dataset_folder
PROJECT_FOLDER      = PARAMS.project_folder

# scan folder
import os
import pandas   as pd
import pydicom

from pathlib                        import Path

def list_dicom_folders():
    dicom_dirs = []

    for dirpath, dirnames, filenames in os.walk(DATASET_FOLDER):
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
    if DATASET_CSV is None:
        path        = Path(DATASET_FOLDER)
        candidates  = list(path.glob('*.nrrd'))
        candidates += list(path.glob('*.nii.gz'))
        candidates += list_dicom_folders()
        dataset     = pd.DataFrame(candidates, columns=['images'])
    else:
        dataset     = pd.read_csv(DATASET_CSV)

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
    PadAndCropTo((192, 192, 160), cval=-1000),
    TransformFromITKFilter(clamp),
    TransformFromITKFilter(cast)
)

# iterate over all images
print ('Loading images...')
dataset = data_inventory()
print ('Loaded {} images'.format(len(dataset)))

for i, row in dataset.iterrows():
    print ('Loading image {} of {}'.format(i+1, len(dataset)))
    try:
        image = loader(row['images'])
    except:
        print ('Error loading image')
        continue

    print ('Saving image...')
    filename = str(i).zfill(12) + ".nii.gz"
    filename = os.path.join(PROJECT_FOLDER, '1.processed', filename + ".nii.gz")
    sitk.WriteImage(image, row['images'] + '.nii.gz')

    print ('Done')