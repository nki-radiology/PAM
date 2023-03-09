import os
import pandas as pd
import numpy as np
import argparse
from datetime                           import datetime

class Preprocessing(object):
    def __init__(self, args):
        self.yads_file_to_read   = args.YADS_file_to_read
        self.min_days_inclusion  = args.min_days_inclusion
        self.max_days_inclusion  = args.max_days_inclusion
        self.path_to_save_file   = args.path_to_save_file
        self.path_to_cts_in_yads = args.path_to_cts_in_yads
        self.patients_file_to_read = args.patients_file_to_read
    
    def assign_path(self, entry):
        path = self.path_to_cts_in_yads + entry.AnonymizedName + '/'
        return path
    
    def assign_number_of_dicoms(self, entry):
        try: 
            subdirs = next(os.walk(entry.PathRawDicom))[1]
            num_subdir = len(subdirs)
        except StopIteration:
            num_subdir = 0
            pass
        return num_subdir
         
    def read_yads_file(self):
        yads_data = pd.read_csv(self.yads_file_to_read)
        print('Initial CT scans number: ', len(yads_data))
        
        yads_data.insert(len(yads_data.columns), 'PathRawDicom', yads_data.apply(self.assign_path, axis=1))
        yads_data.insert(len(yads_data.columns), 'NumberOfDicoms', yads_data.apply(self.assign_number_of_dicoms, axis=1))
        
        # Saving as 1.YADS_with_cts_numbers.csv. This is only to check if the number of dicoms is right
        yads_data.to_csv(self.path_to_save_file + '1.YADS_with_number_of_scans.csv', na_rep='NULL', index=False, encoding='utf-8')
        
        # Removing all the patients with less than 2 scans and saving them
        yads_data = yads_data.loc[yads_data.NumberOfDicoms.apply(lambda x: x >= 2)]
        yads_data.to_csv(self.path_to_save_file + '2.YADS_patients_with_more_than_one_scan.csv', na_rep='NULL', index=False, encoding='utf-8')
        print("End CT scans number: ", len(yads_data))
        return yads_data
    
    
    def get_date(self, directory, idx):
        directory_date           = datetime.strptime(str(directory[idx])[:8], '%Y%m%d').strftime('%Y%m%d')
        remaining_directory_name = str(directory[idx])[8:]
        new_directory            = {'date_directory': directory_date, 'remaining_name_directory': remaining_directory_name}
        return new_directory

    
    def get_difference_days(self, ct1_date, ct2_date):
        input_date_list = [datetime.strptime(ct1_date, '%Y%m%d'), datetime.strptime(ct2_date, '%Y%m%d')]
        difference_days = (input_date_list[1] - input_date_list[0]).days
        return difference_days
           
           
    def get_difference_between_scans(self, yads):
        
        pairs  = []

        for i in range(len(yads)):
            directories      = next(os.walk(yads.iloc[i]['PathRawDicom']))[1]
            directories_date = list(map(lambda i:self.get_date(directories, i), range(0, len(directories))))
            
            # Get the values
            max_len = len(directories_date) - 1
            for j in range (max_len):
                for k in range(j + 1, len(directories_date)):
                    # Sorting dates
                    dates = [directories_date[j], directories_date[k]]
                    dates.sort(key = lambda x: datetime.strptime(x['date_directory'], '%Y%m%d'))
                    # Difference in days
                    difference_in_days_between_scans = self.get_difference_days(dates[0]["date_directory"], dates[1]["date_directory"])
                    pairs.append([ yads.iloc[i]['AnonymizedName'] , yads.iloc[i]['PatientID'], yads.iloc[i]['YADSId'], 
                                  yads.iloc[i]['AnonymizedPatientID'], yads.iloc[i]['Included'], 
                                  dates[0]["date_directory"], yads.iloc[i]['PathRawDicom'] + dates[0]["date_directory"] + dates[0]["remaining_name_directory"], 
                                  dates[1]["date_directory"], yads.iloc[i]['PathRawDicom'] + dates[1]["date_directory"] + dates[1]["remaining_name_directory"],
                                  difference_in_days_between_scans])
 
        yads_including_days = pd.DataFrame(pairs, columns=['AnonymizedName', 'PatientID', 'YADSId', 'AnonymizedPatientID', 'Included', 'PRIOR_DATE', 'PRIOR_PATH', 'SUBSQ_DATE', 'SUBSQ_PATH', 'DifferenceInDaysBetweenScans'])
        yads_including_days.to_csv(self.path_to_save_file + '3.YADS_including_difference_in_days.csv', na_rep='NULL', index=False, encoding='utf-8')
        return yads_including_days

    
    def exclude_patients_based_on_number_of_days(self, data):
        included_patients_yads = data.loc[data.DifferenceInDaysBetweenScans.apply(lambda x: x >= 30 and x <= 120)]
        
        # Changing to date format and avoiding SettingWithCopyWarning
        included_patients_yads = included_patients_yads[included_patients_yads['PRIOR_DATE'].notnull()].copy()
        included_patients_yads = included_patients_yads[included_patients_yads['SUBSQ_DATE'].notnull()].copy()
        included_patients_yads['PRIOR_DATE'] = included_patients_yads['PRIOR_DATE'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))
        included_patients_yads['SUBSQ_DATE'] = included_patients_yads['SUBSQ_DATE'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))
        
        # Sort by AnonymizedName and PRIOR_DATE
        included_patients_yads = included_patients_yads.sort_values(by=['AnonymizedName', 'PRIOR_DATE'])
        
        # Save dataframe
        included_patients_yads.to_csv(args.path_to_save_file + '4.YADS_with_included_patients.csv', na_rep='NULL', index=False, encoding='utf-8')
        return included_patients_yads
    
    
    def read_and_get_included_patients(self, yads_data):
        
        patients_data = pd.read_csv(self.patients_file_to_read)
        patients_data['date_of_death'] = pd.to_datetime(patients_data['date_of_death'], format='%Y-%m-%d', errors='coerce')

        # Rename the patients_data column from avl_id to PatientID to be similar to the column name in yads_data
        patients_data.rename(columns={'avl_id' : 'PatientID'}, inplace = True)
        
        # Selecting date_of_death and date_last_check from the patients file with the same ID as yads_date preprocessed file
        date_of_death   = patients_data.set_index('PatientID')['date_of_death'].to_dict()
        date_last_check = patients_data.set_index('PatientID')['date_last_check'].to_dict()
        yads_data['DateOfDeath']     = yads_data['PatientID'].map(date_of_death)
        yads_data['DateOfLastCheck'] = yads_data['PatientID'].map(date_last_check)
        
        # Saving as 1.YADS_with_cts_numbers.csv. This is only to check if the number of dicoms is right
        yads_data.to_csv(self.path_to_save_file + '5.YADS_including_date_of_death_and_last_check.csv', na_rep='NULL', index=False, encoding='utf-8')
        
        return yads_data
    
        
def main(args):
    
    preproc = Preprocessing(args)
    if args.preprocessing_nki:
        print("Preprocessing YADS file... ")
        yads_data = preproc.read_yads_file()
        yads_including_days = preproc.get_difference_between_scans(yads_data)
        included_patients_yads = preproc.exclude_patients_based_on_number_of_days(yads_including_days)
        print('Done :)')
        
        print('Preprocessing patient file... ')
        new_yads_data = preproc.read_and_get_included_patients(included_patients_yads)
        
    else:
        print('Under development...!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disentanglement Methods: Wassertein Autoencoder, Beta-VAE')
    
    parser.add_argument('--preprocessing_nki',  default=True, type=bool,  help='Preprocessing or ...')
    parser.add_argument('--YADS_file_to_read',  default='/projects/disentanglement_methods/files_nki/YADSRequestResult.csv', type=str, help='YADS file path')
    parser.add_argument('--patients_file_to_read', default='/projects/disentanglement_methods/files_nki/patients.csv', type=str, help='Patients file path')
    parser.add_argument('--min_days_inclusion', default=30,   type=int,   help='Minimum days between prior and subsquent CTs')
    parser.add_argument('--max_days_inclusion', default=120,  type=int,   help='Maximum days between prior and subsquent CTs')
    parser.add_argument('--path_to_save_file',  default='/projects/disentanglement_methods/files_nki/', type=str, help='Path to save the preprocessed file of YADS')
    parser.add_argument('--path_to_cts_in_yads',default='/data/groups/beets-tan/s.trebeschi/INFOa_dicoms/DICOM/', type=str, help='Path to add to the AnonymizedName in yalds to read the CTs')
    
    args = parser.parse_args()
    
    main(args)