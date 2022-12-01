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

    print('Helo----------------------------------------------------------------------------------')
    print(filename)
    # Reading the original csv file
    data               = pd.read_csv(filename,sep=';')
    print(data)
    # Changing for the right path
    data['PRIOR_PATH_NRRD'] = data['PRIOR_PATH_NRRD'].replace('/DATA/laura/external_data/abdomen/', '/IMMUNOTEAM/', regex=True)
    data['SUBSQ_PATH_NRRD'] = data['SUBSQ_PATH_NRRD'].replace('/DATA/laura/external_data/abdomen/', '/IMMUNOTEAM/', regex=True)
    
    # Removing duplicate paths
    #new_prior = data[~data.duplicated('PRIOR_PATH')]
    #new_subsq = data[~data.duplicated('SUBSQ_PATH')]

    # Concatenation of both dataframes and removing duplicate paths
    non_repeating_data = list(data['PRIOR_PATH_NRRD']) + list(data['SUBSQ_PATH_NRRD']) #new_prior
    non_repeating_data = list(dict.fromkeys(non_repeating_data))

    return data, non_repeating_data# new_prior, new_subsq#['PRIOR_PATH'], new_subsq['SUBSQ_PATH'], result


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

    f      = open("/DATA/laura/external_data/unprocessed_external_abdomen_data_2.csv", 'w') 
    writer = csv.writer(f)
    header = ['dicom_path']
    writer.writerow(header)

    # Reading and saving preprocessed data
    with tqdm(total=len(tcia_proc)) as pbar:
        for path in tcia_proc:
            path = path.split('/')[:7]
            print("============================================================")
            print(path)
            path = '/' + path[1] + '/' + path[2] + '/' + path[3] + '/' + path[4] + '/' + path[5] + '/' + path[6]
            print(path + "-------------------------")
            try:
                processed_ct_scan = loader(path)
                processed_ct_scan = processed_ct_scan + 120
                processed_ct_scan = processed_ct_scan / 2
                processed_ct_scan = SimpleITK.Cast(processed_ct_scan, sitk.sitkUInt8)

                path_1 = path.replace("/IMMUNOTEAM", "/DATA/laura/external_data/" + crop)
                verify_path_to_save(path_1)
                ct_path = path_1 + '/' + path.split('/')[6] 
                WriteImage(processed_ct_scan, ct_path)

            except:
                print("--------------- CT was not loaded! ---------------")
                writer.writerow(ct_path)
                pass
            pbar.update(1)
    f.close()
    print("Done!")


path                     = "/DATA/laura/external_data/thorax/features_pam_thorax_unprocessed_nrrd.csv"
data, non_repeating_data = get_data_file(path)
print(len(non_repeating_data))
apply_localizer(non_repeating_data, 'thorax')
