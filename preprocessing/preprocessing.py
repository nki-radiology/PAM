import os
import numpy         as     np
import pandas        as     pd
from   pydicom       import dcmread
from   tqdm.notebook import tqdm


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


def get_data_file(filename):
    tcia = pd.read_csv(filename, index_col=0)
    print('Initial CT scans number: ', len(tcia))
    tcia.insert(len(tcia.columns), 'patient_id', tcia.index)
    tcia.insert(len(tcia.columns), 'fold', tcia.apply(assign, axis=1))

    tcia = tcia.loc[
           tcia.dicom_path.apply(lambda x: ('CancerImagingArchive_20200421/CT' in x)) &
           tcia.apply(lambda x: safe_difference(x.first_axial_coord, x.second_axial_coord) >= 0.1, axis=1) &
           tcia.apply(lambda x: safe_difference(x.first_axial_coord, x.second_axial_coord) <= 5.0, axis=1) &
           tcia.number_of_slices.apply(lambda x: x >= 32), :] # Create a variable with slices number and fold name

    return tcia


def create_folder_to_save(root_folder, son_folders):
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    for son_folder in son_folders:
        if not os.path.exists(os.path.join(root_folder, son_folder)):
            os.makedirs(os.path.join(root_folder, son_folder))


def clipp_HU_and_save_data(data_file, path_data_to_process , path_to_save):
    log   = []
    count = []
    for entry_id, entry in tqdm(data_file.iterrows(), total=data_file.shape[0]):
        try:
            for i in range(5): # Why 5?
                count = entry_id * 10 + i
                np.random.seed(count)

                path_dicom   = path_data_to_process  + entry.dicom_path
                dcm_file     = sample(path_dicom)
                img_1, loc_1 = read(dcm_file)
                print("Shape 1: ", img_1.shape, loc_1)

                path_dicom   = path_data_to_process  + entry.dicom_path
                dcm_file     = sample(path_dicom)
                img_2, loc_2 = read(dcm_file)
                print("Shape 2: ", img_2.shape, loc_2)

                if (img_1.std() < 0.1) or (img_2.std() < 0.1):
                    continue

                y     = loc_1 > loc_2
                img_1 = np.clip(img_1, -120, 300) + 120
                img_2 = np.clip(img_2, -120, 300) + 120

                # Padding
                xxy = np.stack([img_1, img_2, np.zeros_like(img_1) + y], axis=-1)
                xxy = xxy.astype(np.uint16)
                print("My final shape: ", xxy.shape)

                path = os.path.join(path_to_save, entry.fold)
                with open(path + "/%08d" % count + '.npy', 'wb') as f:
                    np.save(f, xxy)

        except:
            log.append([entry_id, i, count])


data = get_data_file("../../Data/tcia_dataset.csv")
print('Preprocessed CT scans number: ', len(data))
#print(data.columns)
#print(data.head)

path_data_to_process = '../../../../..'
path_root_to_save = '../../../../../DATA/laura/tcia_temp'
fold_to_save_data = ['train', 'test']
create_folder_to_save(path_root_to_save, fold_to_save_data)
clipp_HU_and_save_data(data[0:5], path_data_to_process, path_root_to_save)