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
from   localizer import *
import argparse


class Pairs_Format(object):
    def __init__(self, args):
        
        self.pairs_file    = args.pairs_file
        self.path_of_dicom = args.path_of_dicom
        self.path_of_nrrd  = args.path_of_nrrd
        self.structure_to_crop              = args.structure_to_crop
        self.path_to_save_unprocessed_pairs = args.path_to_save_unprocessed_pairs
        

    def get_data_file(self):
        
        # Reading the original csv file
        data               = pd.read_csv(self.pairs_file)
        print(' Initial data length: ', len(data))
        
        # Removing duplicate paths
        new_prior = data[~data.duplicated('PRIOR_PATH')]
        new_subsq = data[~data.duplicated('SUBSQ_PATH')] 
        print('Non-repeated prior data length: ', len(new_prior))
        print('Non-repeated subsq data length: ', len(new_subsq))

        # Concatenation of both dataframes and removing duplicate paths
        non_repeating_data = list(new_prior['PRIOR_PATH']) + list(new_subsq['SUBSQ_PATH'])
        print('Non-repeated total data length: ', len(non_repeating_data))
        non_repeating_data = list(dict.fromkeys(non_repeating_data))
        print(' Total len of non-repeated data PRIOR_PATH plus SUBSQ_PATH: ', len(non_repeating_data))
        non_repeating_pairs = pd.DataFrame(non_repeating_data, columns=[ ' Path ' ])
        non_repeating_pairs.to_csv(self.path_to_save_unprocessed_pairs + 'non_reapiting_pairs_prior_plus_subsq.csv', na_rep='NULL', index=False, encoding='utf-8')
        return non_repeating_data


    def verify_path_to_save(self, path: str):
        if not os.path.exists(path):
            print('Creating folder...')
            os.makedirs(path)
        else:
            print('This folder already exists :)!')


    def apply_localizer(self, data: list): 
        clamp = ClampImageFilter()
        clamp.SetUpperBound(300)
        clamp.SetLowerBound(-120)

        if self.structure_to_crop == 'thorax': 
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

        name_unprocessed_file  = self.path_to_save_unprocessed_pairs + self.structure_to_crop + "_data.csv"
        dicom_unprocessed_path = []
        
        # Reading and saving preprocessed data
        with tqdm(total=len(data)) as pbar:
            for path in data:
                try:
                    processed_ct_scan = loader(path)
                    processed_ct_scan = processed_ct_scan + 120
                    processed_ct_scan = processed_ct_scan / 2
                    processed_ct_scan = SimpleITK.Cast(processed_ct_scan, sitk.sitkUInt8)

                    path_to_save_nrrd = path.replace(self.path_of_dicom, self.path_of_nrrd + self.structure_to_crop )
                    print('Path to save: ', path_to_save_nrrd)
                    self.verify_path_to_save(path_to_save_nrrd)
                    nrrd_image_name = path_to_save_nrrd + '/' + path.split('/')[9] + '.nrrd'
                    print('Nrrd image name: ', nrrd_image_name)
                    WriteImage(processed_ct_scan, nrrd_image_name)
                except:
                    print("--------------- CT was not loaded! ---------------")
                    dicom_unprocessed_path.append(path)
                    pass
                pbar.update(1)
        print("Done!")
        dict = {'dicom_path': dicom_unprocessed_path}
        df   = pd.DataFrame(dict)
        df.to_csv(name_unprocessed_file, na_rep='NULL', index=False, encoding='utf-8')


def main(args):
    dcm_to_nrrd = Pairs_Format(args)
    non_repeated_data = dcm_to_nrrd.get_data_file()
    dcm_to_nrrd.apply_localizer(non_repeated_data)
    print('Process Done!!!!!!!!!!!!!!!!!!!!!!!')    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='From Dicom to Nrrd class for all CT pairs')
    parser.add_argument('--structure_to_crop',              default='abdomen',                                                     type=str, help='Structure to crop: thorax or abdomen')
    parser.add_argument('--pairs_file',                     default='/projects/disentanglement_methods/files_nki/infoA/pairs.csv', type=str, help='Pairs file path')
    parser.add_argument('--path_of_dicom',                  default='/data/groups/beets-tan/s.trebeschi/INFOa_dicoms/DICOM',       type=str, help='Dicom path')
    parser.add_argument('--path_of_nrrd',                   default='/data/groups/beets-tan/l.estacio/infoA/',                     type=str, help='Nrrd path')
    parser.add_argument('--path_to_save_unprocessed_pairs', default='/projects/disentanglement_methods/files_nki/infoA/',          type=str, help='Path to save unprocessed pairs')
    
    args = parser.parse_args()
    
    main(args)
        