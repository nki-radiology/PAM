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
from   Localizer           import *
from   config              import args_localizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    tcia = pd.read_csv(filename)
    print('Initial CT scans number: ', len(tcia))
    tcia.insert(len(tcia.columns), 'patient_id', tcia.index)

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


def apply_localizer(tcia_proc: str, path_to_save: str, root_path='', non_proc_tcia=''):

    clamp = ClampImageFilter()
    clamp.SetUpperBound(300)
    clamp.SetLowerBound(-120)

    loader = ImageLoader(
        ReadDICOM(),
        CropAbdomen(margin=25), # CropThorax(margin=25),#(margin=5),
        Resample(2),
        PadAndCropTo((192, 192, 160), cval=-1000),
        TransformFromITKFilter(clamp),
    )

    f      = open(path_to_save + non_proc_tcia, 'w') # ('unprocessed_cts_2022.csv', 'w')
    writer = csv.writer(f)
    header = ['dicom_path', 'first_axial_coord', 'number_of_slices', 'second_axial_coord', 'patient_id']
    writer.writerow(header)

    # Reading and saving preprocessed data
    with tqdm(total=len(tcia_proc)) as pbar:
        for index, row in tcia_proc.iterrows():
            try:
                path              = root_path + row['dicom_path']
                processed_ct_scan = loader(path)

                # The 120 value is to compensate the clamp.SetLowerBound?
                processed_ct_scan = processed_ct_scan + 120
                processed_ct_scan = processed_ct_scan / 2
                processed_ct_scan = SimpleITK.Cast(processed_ct_scan, sitk.sitkUInt8)

                ct_path = path_to_save + str(index) + '_' + str(row['dicom_path'].split('/')[5]) + '.nrrd'
                WriteImage(processed_ct_scan, ct_path)

            except:
                print("--------------- CT was not loaded! ---------------")
                writer.writerow(row)
                pass
            pbar.update(1)
    f.close()



def process_tcia_file():
    # Verifying the existence of the folder to save the processed CTs and files
    verify_path_to_save(args_localizer.path_to_save_proc_cts)

    # Obtaining the file paths that contains all the CTs to be processed
    proc_tcia = get_data_file(args_localizer.inp_tcia_path, args_localizer.inp_cts_path)

    # Saving the new TCIA file (processed)
    save_process_tcia_file(proc_tcia, args_localizer.path_to_save_proc_cts, args_localizer.proc_tcia_file_name)


def process_cts():
    # Reading the CTs according to the new TCIA files (processed)
    name_tcia = args_localizer.path_to_save_proc_cts + args_localizer.proc_tcia_file_name
    proc_tcia = pd.read_csv(name_tcia, index_col=0)

    # Applying the localizer
    apply_localizer(tcia_proc=proc_tcia, path_to_save=args_localizer.path_to_save_proc_cts,
                    root_path=args_localizer.root_path_to_add, non_proc_tcia=args_localizer.non_proc_tcia_file_name)


def start_preprocessing():
    # Step 1: preprocess the tcia file
    # process_tcia_file()

    # Step 2: preprocess the output file of the step 1
    # process_cts()
    print("End Preprocessing :) ")


start_preprocessing()

