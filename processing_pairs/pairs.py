import os
import pandas as pd
import argparse
import pydicom as dicom
from datetime import datetime
from dateutil.parser import parse
from preprocessing   import Preprocessing
import SimpleITK as sitk

class Pairs(object):
    def __init__(self, args):
        self.name_file_to_read           = args.name_file_to_read
        self.name_pairs_file             = args.name_pairs_file


    def get_directories(self, root_path):
        
        try: 
            subdirs = next(os.walk(root_path))[1]
            subdirs = list(map(lambda x: root_path + '/' + x, subdirs))
        except StopIteration:
            subdirs = []
            pass
        
        return subdirs
    


    def get_pairs(self):
        yads  = pd.read_csv(self.name_file_to_read)
        pairs = []
        for i in range(len(yads)):
            # Get directories with possible dicoms
            prior_subdirs = self.get_directories(yads.iloc[i]['PRIOR_PATH'])
            subsq_subdirs = self.get_directories(yads.iloc[i]['SUBSQ_PATH'])
            
            for prior in prior_subdirs:
                for subsq in subsq_subdirs:
                    pairs.append([ yads.iloc[i]['AnonymizedName'] , yads.iloc[i]['PatientID'], yads.iloc[i]['YADSId'], yads.iloc[i]['AnonymizedPatientID'], yads.iloc[i]['Included'], 
                                   yads.iloc[i]['PRIOR_DATE'], prior,
                                   yads.iloc[i]['SUBSQ_DATE'], subsq,
                                   yads.iloc[i]['DifferenceInDaysBetweenScans'], yads.iloc[i]['DateOfDeath'], yads.iloc[i]['DateOfLastCheck'], yads.iloc[i]['DaysOfSurvival'],
                                   yads.iloc[i]['Event'], yads.iloc[i]['Y1Survival'], yads.iloc[i]['Y2Survival']
                                ])
        yads_pairs = pd.DataFrame(pairs, columns=[ 'AnonymizedName', 'PatientID', 'YADSId', 'AnonymizedPatientID', 'Included', 'PRIOR_DATE', 'PRIOR_PATH',
                                                    'SUBSQ_DATE', 'SUBSQ_PATH', 'DifferenceInDaysBetweenScans', 'DateOfDeath', 'DateOfLastCheck', 'DaysOfSurvival',
                                                    'Event', 'Y1Survival', 'Y2Survival'
                                                 ])
        yads_pairs.to_csv(self.name_pairs_file, na_rep='NULL', index=False, encoding='utf-8')
                    
        
def main(args):
    
    if args.preprocessing_nki:
        
        preproc = Preprocessing(args)
        
        print("Preprocessing YADS file... ")
        yads_data = preproc.read_yads_file()
        yads_including_days = preproc.get_difference_between_scans(yads_data)
        included_patients_yads = preproc.exclude_patients_based_on_number_of_days(yads_including_days)
        print('Done :)')
        
        print('Preprocessing patient file... ')
        yads_including_dates_from_patients = preproc.read_and_get_included_patients(included_patients_yads)
        yads_including_survival            = preproc.get_survival_from_patients(yads_including_dates_from_patients)
        print('Done :)')
        print('--------------------------------------')
        print('Applying the last fomat to data ......')
        preproc.final_format_dates(yads_including_survival)
        print('--------------------------------------')
        print('--------------------------------------')
        print('End Survival Data Preproprocessing! :)')
        
    else:
        pairs = Pairs(args)
        print(' Generating pairs file ....')
        pairs.get_pairs()
        print('End Generating pairs file! :)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing of YADS and patient files to create pairs')
    # Variables required for preprocessing the pairs
    parser.add_argument('--preprocessing_nki',  default=False, type=bool,  help='Preprocessing or ...')
    parser.add_argument('--YADS_file_to_read',  default='/projects/disentanglement_methods/files_nki/YADSRequestResult.csv', type=str, help='YADS file path')
    parser.add_argument('--patients_file_to_read', default='/projects/disentanglement_methods/files_nki/patients.csv', type=str, help='Patients file path')
    parser.add_argument('--min_days_inclusion', default=30,   type=int,   help='Minimum days between prior and subsquent CTs')
    parser.add_argument('--max_days_inclusion', default=120,  type=int,   help='Maximum days between prior and subsquent CTs')
    parser.add_argument('--y1_survival_days',   default=355,  type=int,   help='Number of days to assign survival')
    parser.add_argument('--y2_survival_days',   default=710,  type=int,   help='Number of days to assign survival')
    parser.add_argument('--path_to_save_file',  default='/projects/disentanglement_methods/files_nki/', type=str, help='Path to save the preprocessed file of YADS')
    parser.add_argument('--path_to_cts_in_yads',default='/data/groups/beets-tan/s.trebeschi/INFOa_dicoms/DICOM/', type=str, help='Path to add to the AnonymizedName in yalds to read the CTs')
    
    # Variables requires for creating the pairs
    parser.add_argument('--name_file_to_read',           default='/projects/disentanglement_methods/files_nki/7.YADS_survival_data_standardized.csv', type=str, help='File which contains initial pairs')
    parser.add_argument('--name_pairs_file',             default='/projects/disentanglement_methods/files_nki/infoA/pairs.csv', type=str, help='Path to save file with all the pairs for infoA')
    args = parser.parse_args()
    
    main(args)