import os
import csv
import SimpleITK
import numpy     as     np
import pandas    as     pd
import SimpleITK as     sitk
from   tqdm      import tqdm
from   pydicom   import dcmread
from   SimpleITK import WriteImage
from   SimpleITK import ClampImageFilter
from   Localizer import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_data_file(filename: str):

    # Reading the original csv file
    data               = pd.read_csv(filename)

    # Changing for the right path
    data['PRIOR_PATH'] = data['PRIOR_PATH'].replace('Z:', '/IMMUNOTEAM', regex=True)
    data['SUBSQ_PATH'] = data['SUBSQ_PATH'].replace('Z:', '/IMMUNOTEAM', regex=True)
    data['PRIOR_PATH'] = data['PRIOR_PATH'].replace(r'\\', '/',  regex=True)
    data['SUBSQ_PATH'] = data['SUBSQ_PATH'].replace(r'\\', '/', regex=True)

    # Removing duplicate paths
    new_prior = data[~data.duplicated('PRIOR_PATH')]
    new_subsq = data[~data.duplicated('SUBSQ_PATH')] 

    # Concatenation of both dataframes and removing duplicate paths
    non_repeating_data = list(new_prior['PRIOR_PATH']) + list(new_subsq['SUBSQ_PATH'])
    print(len(non_repeating_data))

    return data, non_repeating_data


def verify_path_to_save(path: str):
    if not os.path.exists(path):
        print('Creating folder...')
        os.makedirs(path)
    else:
        print('This folder already exists :)!')


def apply_localizer(tcia_proc: list, crop: str): 
    clamp = ClampImageFilter()
    clamp.SetUpperBound(300)
    clamp.SetLowerBound(-120)

    if crop == 'thorax': 
        loader = ImageLoader(
            ReadDICOM(),
            CropThorax(margin=25),
            Resample(2),
            PadAndCropTo((192, 192, 160), cval=-1000),
            TransformFromITKFilter(clamp),
        )
        print('Applying crop of Thorax!')
    
    else:
        loader = ImageLoader(
            ReadDICOM(),
            CropAbdomen(margin=25), 
            Resample(2),
            PadAndCropTo((192, 192, 160), cval=-1000),
            TransformFromITKFilter(clamp),
        )
        print('Applying crop of Abdomen!')

    name_f = "/DATA/laura/data_melda/unprocessed_" + crop + "_data.csv"
    f      = open(name_f, 'w') 
    writer = csv.writer(f)
    header = ['dicom_path']
    writer.writerow(header)

    # Reading and saving preprocessed data
    with tqdm(total=len(tcia_proc)) as pbar:
        for path in tcia_proc:
            print(' My Path is: ---------------------------------')
            print(path)
            try:
                processed_ct_scan = loader(path)
                processed_ct_scan = processed_ct_scan + 120
                processed_ct_scan = processed_ct_scan / 2
                processed_ct_scan = SimpleITK.Cast(processed_ct_scan, sitk.sitkUInt8)

                path_1 = path.replace("/IMMUNOTEAM", "/DATA/laura/data_melda/" + crop)
                verify_path_to_save(path_1)
                ct_path = path_1 + '/' + path.split('/')[6] + '.nrrd'
                print(' My New Path is: ---------------------------------')
                print(ct_path)
                WriteImage(processed_ct_scan, ct_path)

            except:
                print("--------------- CT was not loaded! ---------------")
                writer.writerow(path)
                pass
            pbar.update(1)
    f.close()
    print("Done!")


path                     = "/SHARED/active_Laura/Features_Melda/ScanPairs.csv"
data, non_repeating_data = get_data_file(path)
apply_localizer(non_repeating_data, 'abdomen')

