import os
import csv
import SimpleITK
import numpy               as     np
import pandas              as     pd
import SimpleITK           as     sitk
from   tqdm                import tqdm
from   pydicom             import dcmread
from   SimpleITK           import WriteImage
from   SimpleITK           import ClampImageFilter
from   SimpleITK           import GetImageFromArray
from   inference_localizer import *

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def read(filename):
    image = dcmread(filename)
    coord = float(image.ImagePositionPatient[-1])
    b     = float(image.RescaleIntercept)
    m     = float(image.RescaleSlope)
    image = image.pixel_array * m + b
    return image, coord


def sample(dicom_dir):
    dcm_files     = os.listdir(dicom_dir)
    rnd_dcm_files = np.random.choice(dcm_files)
    rnd_dcm_files = os.path.join(dicom_dir, rnd_dcm_files)
    return rnd_dcm_files


def safe_difference(a, b):
    try:
        return np.abs(a-b)
    except:
        return 0


def  assign(entry):
    if 'TCGA' in entry.dicom_path:
        return 'val'
    elif entry.patient_id % 10:
        return 'train'
    return 'test'


def get_data_file(filename: str, path_to_select: str):
    tcia = pd.read_csv(filename, index_col=0)
    print('Initial CT scans number: ', len(tcia))
    tcia.insert(len(tcia.columns), 'patient_id', tcia.index)
    #tcia.insert(len(tcia.columns), 'fold', tcia.apply(assign, axis=1))

    tcia = tcia.loc[
           tcia.dicom_path.apply(lambda x: (path_to_select in x)) &
           tcia.apply(lambda x: safe_difference(x.first_axial_coord, x.second_axial_coord) >= 0.1, axis=1) &
           tcia.apply(lambda x: safe_difference(x.first_axial_coord, x.second_axial_coord) <= 5.0, axis=1) &
           tcia.number_of_slices.apply(lambda x: x >= 50), :]

    return tcia


def verify_path_to_save(path: str):
    if not os.path.exists(path):
        print('Creating folder...')
        os.makedirs(path)
    else:
        print('This folder already exists :)!')


def save_process_tcia_file(tcia, path_to_save, proc_file_name):
    tcia.to_csv(path_to_save + proc_file_name)
    print("The file was processed and saved!")



def apply_localizer(tcia_proc: str, path_to_save: str, root_path=''):

    clamp = ClampImageFilter()
    clamp.SetUpperBound(300)
    clamp.SetLowerBound(-120)

    loader = ImageLoader(
    	ReadDICOM(),
    	CropThorax(margin=5),
        Resample(2),
        PadAndCropTo((192, 192, 160), cval=-1000),
        TransformFromITKFilter(clamp),
    )

    f      = open('unprocessed_cts.csv', 'w')
    writer = csv.writer(f)
    header = ['dicom_path', 'first_axial_coord', 'number_of_slices', 'second_axial_coord', 'patient_id']
    writer.writerow(header)

    # Reading and saving preprocessed data
    with tqdm(total=len(tcia_proc)) as pbar:
        for index, row in tcia.iterrows():
            try:
                path              = root_path + row['dicom_path']
                processed_ct_scan = loader(path)

                # The 120 value is to compensate the clamp.SetLowerBound?
                processed_ct_scan = processed_ct_scan + 120
                processed_ct_scan = processed_ct_scan / 2
                processed_ct_scan = SimpleITK.Cast(processed_ct_scan,sitk.sitkUInt8)

                ct_path = path_to_save + str(index) + '_' + str(row['dicom_path'].split('/')[5]) +  '.nrrd'
                WriteImage(processed_ct_scan, ct_path)

            except:
                print("--------------- CT was not loaded! ---------------")
                writer.writerow(row)
                pass
            pbar.update(1)
    f.close()


# ------------------------- Save the preprocessed csv file -------------------------
'''
path_tcia_csv  = "../../Data/tcia_dataset.csv"
path_to_select = '/IMMUNOTEAM/CancerImagingArchive_20200421/CT/CT Lymph Nodes/'
path_to_save   = '../../../../../DATA/laura/tcia_proc_data/CT_Lymph_Nodes/'
proc_file_name = 'tcia_ct_lymph_nodes.csv'

tcia = get_data_file(path_tcia_csv, path_to_select)
verify_path_to_save(path_to_save)
save_process_tcia_file(tcia, path_to_save, proc_file_name)
'''
# ----------------------------------------------------------------------------------


# ------------------------ Save the preprocessed nrrd images ------------------------
'''
tcia_file_path       = "../../Data/tcia_ct_lymph_nodes.csv"
path_root_to_save    = '../../../../../DATA/laura/tcia_proc_data/CT_Lymph_Nodes/'
path_data_to_process = '../../../../..'
tcia                 = pd.read_csv(tcia_file_path, index_col=0)
apply_localizer(tcia_proc = tcia, path_to_save = path_root_to_save,
                            root_path=path_data_to_process)
'''
# ----------------------------------------------------------------------------------

