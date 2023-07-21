
import pandas as pd
import os
import argparse
import pydicom

from tqdm import tqdm

import SimpleITK as sitk

from SimpleITK import ClampImageFilter
from SimpleITK import CastImageFilter
from SimpleITK import WriteImage

# import local libraries 
import sys
sys.path.append('../../')

from libs.frida.io                  import ReadDICOM
from libs.frida.io                  import ImageLoader

from libs.frida.transforms          import TransformFromITKFilter
from libs.frida.transforms          import Resample
from libs.frida.transforms          import PadAndCropTo

from Localizer                      import CropAbdomen  
from Localizer                      import CropThorax

# parse arguments
argparser = argparse.ArgumentParser()

argparser.add_argument("--data-folder",     type=str, default="/data")
argparser.add_argument("--body-part",       type=str, default="thorax")
argparser.add_argument("--output-folder",   type=str, default="/output")

args = argparser.parse_args()

data_folder     = args.data_folder
body_part       = args.body_part
output_folder   = args.output_folder

# define transforms
CropObj = CropThorax if body_part == "thorax" else CropAbdomen

clamp = ClampImageFilter()
clamp.SetUpperBound(300)
clamp.SetLowerBound(-120)

cast = CastImageFilter()
cast.SetOutputPixelType(sitk.sitkInt16)

loader = ImageLoader(
    ReadDICOM(),
    CropObj(margin=25),
    Resample(2),
    PadAndCropTo((192, 192, 160), cval=-1000),
    TransformFromITKFilter(clamp),
    TransformFromITKFilter(cast)
)

# list folders 
def list_subdirectories_with_no_subdirectory(directory):
    subdirectories_with_no_subdir = []
    
    for root, dirs, files in os.walk(directory):
        if not dirs:  # Check if the current directory has no subdirectories
            subdirectories_with_no_subdir.append(root)
    
    return subdirectories_with_no_subdir


dcm_folders = list_subdirectories_with_no_subdirectory(data_folder)

# check if ct scans
def is_ct_scan(dicom_file):
    try:
        ds = pydicom.dcmread(dicom_file)
        modality = ds.get("Modality", "").upper()
        if modality == "CT":
            return True
        else:
            return False
    except pydicom.errors.InvalidDicomError:
        return False

def all_files_are_ct_scans(dicom_folder):
    for root, _, files in os.walk(dicom_folder):
        for file in files:
            dicom_file = os.path.join(root, file)
            if not is_ct_scan(dicom_file):
                return False
    return True


# load images
df = []
with tqdm(total=len(dcm_folders)) as pbar:
    for ix, dcm_folder in enumerate(dcm_folders):
        try:
            # check if all files are CT scans
            if not all_files_are_ct_scans(dcm_folder):
                raise Exception("Not all files are CT scans")
            
            #import pdb; pdb.set_trace()
            #break_loop = False
            #if break_loop:
            #    break

            # load image
            image = loader(dcm_folder)
            
            # save image
            filename = str(ix).zfill(12) + ".nii.gz"
            filename = os.path.join(output_folder, filename + ".nii.gz")
            WriteImage(image, filename)

        except:
            filename = -1

        finally:
            df.append({
                "input": dcm_folder,
                "output": filename
            })
                
            # update progress bar
            pbar.update(1)

pd.DataFrame(df).to_csv(os.path.join(output_folder, "preprocessing.csv"), index=False)