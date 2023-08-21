

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input',              type = str, default = None,             help = 'folder that contains the dataset')
parser.add_argument('--output',             type = str, default = None,             help = 'folder that contains the dataset')

parser.add_argument('--debug' ,             type = bool, default = False,           help = 'debug mode')

PARAMS                                      = parser.parse_args()

INPUT               = PARAMS.input
OUTPUT              = PARAMS.output

DEBUG               = PARAMS.debug

####

import os
import pandas               as pd
import pydicom

from pathlib                import Path


from SimpleITK              import ReadImage
from SimpleITK              import ImageSeriesReader
from SimpleITK              import WriteImage
from SimpleITK              import LabelShapeStatisticsImageFilter



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


def log_dataset(dataset):
    filepath = os.path.join(OUTPUT, 'dataset.csv')

    if os.path.isfile(filepath):
        dataset_old = pd.read_csv(filepath)
        dataset = dataset[~dataset.isin(dataset_old.to_dict(orient='list')).all(axis=1)]
        dataset.to_csv(filepath, mode='a', header=False, index=False)
        print(' -- dataset updated')

    else:
        dataset.to_csv(filepath, index=False)
        print(' -- dataset saved')

    return dataset
    

def read_image(path):
    image = None

    try: 
        if os.path.isdir(path):
            reader      = ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dicom_names)
            image       = reader.Execute()
        elif os.path.isfile(path):
            image       = ReadImage(path)
        else:
            pass
    except:
        pass

    return image
    

def segment(dataset):
    for i, row in dataset.iterrows():
        # skip if output already exists
        compressed_filepath_output = os.path.join(OUTPUT, str(i).zfill(12) + '.seg.nii.gz')
        if os.path.isfile(compressed_filepath_output):
            print(' -- skipping', str(i), 'out of', len(dataset), '\t', row['images'])
            continue

        # read image
        print('processing', str(i), 'out of', len(dataset), '\t', row['images'])
        image_path      = row['images']
        image           = read_image(image_path)

        if image is None:
            print(' -- something went wrong reading ' + image_path)
            continue

        # write temp image
        filepath_input  = os.path.join(OUTPUT, 'temp-input.nii')
        WriteImage(image, filepath_input)

        # run total segmentor
        filepath_output = os.path.join(OUTPUT, 'temp-output.nii')
        cmd = 'TotalSegmentator --ml --fast -i ' + filepath_input + ' -o ' +  filepath_output

        try:
            os.system(cmd)
        except:
            print(' -- something went wrong segmenting ' + image_path)
            continue

        # compress output
        print(' -- compressing output')
        image = ReadImage(filepath_output)
        WriteImage(image, compressed_filepath_output)

    os.remove(filepath_input)
    os.remove(filepath_output)


def compute_volumes():

    path            = Path(OUTPUT)
    segmentations   = list(path.glob('*.seg.nii.gz'))

    stats           = LabelShapeStatisticsImageFilter()
    df              = []

    for segmentation in segmentations:
        print('processing', segmentation.name)
        segmentation    = ReadImage(str(segmentation))
        stats.Execute(segmentation)
 
        for label in stats.GetLabels():
            entry = dict()
            entry['filename']   = segmentation.name
            entry['label']      = label
            entry['volume_mL']  = stats.GetPhysicalSize(label) / 1000.
            entry['on_border']  = stats.GetPerimeterOnBorder(label) != 0
            df.append(entry)

    df = pd.DataFrame(df).to_csv(os.path.join(OUTPUT, 'segmentation_volumes.csv'), index=False)


if __name__ == '__main__':

    if DEBUG:
        import pdb; pdb.set_trace()

    print('Parameters:')
    print(PARAMS)

    print('\n#\nRunning data inventory... (might take a while)')

    dataset = data_inventory()

    print('Data inventory:')
    print(' -- total images:',                  len(dataset))
    print(' -- total number *.nrrd:',           len(dataset[dataset['images'].str.endswith('.nrrd')]))
    print(' -- total number *.nii.gz:',         len(dataset[dataset['images'].str.endswith('.nii.gz')]))
    print(' -- total number DICOM folders:',    len(dataset[dataset['images'].apply(lambda x: os.path.isdir(x))]))

    dataset = log_dataset(dataset)

    print('\n#\nRunning segmentation... (might take a while)')

    segment(dataset)

    print('\n#\nRunning volume computation... (might take a while)')

    compute_volumes()


