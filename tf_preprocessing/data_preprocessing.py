import os
import numpy   as     np
import pandas  as     pd
from   tqdm    import tqdm
from   pydicom import dcmread
from SimpleITK import WriteImage
from SimpleITK import ClampImageFilter
from SimpleITK import GetImageFromArray
from inference_localizer import *


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
           tcia.number_of_slices.apply(lambda x: x >= 50), :] # Create a variable with slices number and fold name

    return tcia


def verify_path_to_save(path: str):
    if not os.path.exists(path):
        print('Creating folder...')
        os.makedirs(path)
    else:
        print('This folder already exists :)!')


def save_process_tcia_file(tcia, path_to_save, proc_file_name):
    tcia.to_csv(path_to_save + proc_file_name)
    print("My end shape: ", len(tcia))
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
        ZeroOneScaling(),
        ToNumpyArray(add_batch_dim=False, add_singleton_dim=False)
    )


    # Reading and saving preprocessed data
    with tqdm(total=len(tcia_proc)) as pbar:
        for index, row in tcia.iterrows():
            try:
                print("This is my path: ", row['image_path'])
                path = root_path + row['image_path']
                processed_ct_scan = loader(path)#(row['image_path'])
                processed_ct_scan = GetImageFromArray(processed_ct_scan)

                WriteImage(processed_ct_scan, path_to_save + str(index) + '.nrrd')

            except:
                print("---------- CT was not loaded! ----------")
                pass
            pbar.update(1)


#path_tcia_csv  = "../../Data/tcia_dataset.csv"
#path_to_select = '/IMMUNOTEAM/CancerImagingArchive_20200421/CT/CT Lymph Nodes/'
#path_to_save   = '../../../../../DATA/laura/tcia_proc_data/CT_Lymph_Nodes/' #'../../Data/'
#proc_file_name = 'tcia_ct_lymph_nodes.csv'

#tcia = get_data_file(path_tcia_csv, path_to_select)
#verify_path_to_save(path_to_save)
#save_process_tcia_file(tcia, path_to_save, proc_file_name)
#print('Thank you so much dear God :)!')





tcia_file_path       = "../../Data/tcia_ct_lymph_nodes.csv"
path_root_to_save    = '../../temporal/'
path_data_to_process = '../../../../..'
tcia                 = pd.read_csv(tcia_file_path, index_col=0)
tcia                 = tcia[0:3]


apply_localizer(tcia_proc= tcia, path_to_save= path_root_to_save, root_path=path_data_to_process)


