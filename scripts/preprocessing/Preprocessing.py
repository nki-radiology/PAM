
import os
import argparse

from tqdm import tqdm

import SimpleITK as sitk

from SimpleITK import ClampImageFilter
from SimpleITK import AddImageFilter
from SimpleITK import CastImageFilter

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
    CropObj,
    Resample(2),
    PadAndCropTo((192, 192, 160), cval=-1000),
    TransformFromITKFilter(clamp),
    TransformFromITKFilter(cast)
)


import pdb
pdb.set_trace()

# list folders 
def list_subdirectories_with_no_subdirectory(directory):
    subdirectories_with_no_subdir = []
    
    for root, dirs, files in os.walk(directory):
        if not dirs:  # Check if the current directory has no subdirectories
            subdirectories_with_no_subdir.append(root)
    
    return subdirectories_with_no_subdir


dcm_folders = list_subdirectories_with_no_subdirectory(data_folder)

# load images
df = []
with tqdm(total=len(dcm_folders)) as pbar:
    for ix, dcm_folder in enumerate(dcm_folders):
        try:
            # load image
            image = loader(dcm_folder)
            
            # save image
            filename = str(ix).zfill(12) + ".nii.gz"
            filename = os.path.join(output_folder, filename + ".nii.gz")
            sitk.WriteImage(image, filename)

        except:
            filename = -1

        df.append({
            "input": dcm_folder,
            "output": filename
        })
            
        # update progress bar
        pbar.update(1)