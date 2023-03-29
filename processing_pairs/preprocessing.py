import os
import pandas as pd
import numpy as np
import argparse
import math
from datetime import datetime
from dateutil.parser import parse

class Preprocessing(object):
    def __init__(self, args):
        self.yads_file_to_read   = args.YADS_file_to_read
        self.min_days_inclusion  = args.min_days_inclusion
        self.max_days_inclusion  = args.max_days_inclusion
        self.y1_survival_days    = args.y1_survival_days
        self.y2_survival_days    = args.y2_survival_days
        self.path_to_save_file   = args.path_to_save_file
        self.path_to_cts_in_yads = args.path_to_cts_in_yads
        self.patients_file_to_read = args.patients_file_to_read
        
        # Variables for the pairs file generation
        self.name_file_to_read  = args.name_file_to_read
        self.name_pairs_file    = args.name_pairs_file
    
    
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
        # Assigning the proper path of scans, number of scans, and saving it as csv file.
        yads_data.insert(len(yads_data.columns), 'PathRawDicom', yads_data.apply(self.assign_path, axis=1))
        yads_data.insert(len(yads_data.columns), 'NumberOfDicoms', yads_data.apply(self.assign_number_of_dicoms, axis=1))
        yads_data.to_csv(self.path_to_save_file + '1.YADS_with_number_of_scans.csv', na_rep='NULL', index=False, encoding='utf-8')
        # Removing all the patients with less than 2 scans and saving them as csv file
        yads_data = yads_data.loc[yads_data.NumberOfDicoms.apply(lambda x: x >= 2)]
        yads_data.to_csv(self.path_to_save_file + '2.YADS_patients_with_more_than_one_scan.csv', na_rep='NULL', index=False, encoding='utf-8')
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
        pairs = []
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
        # Create a copy to avoid SettingWithCopyWarning to sort dataframe based on AnonymizedName, and PRIOR_DATE
        included_patients_yads = included_patients_yads[included_patients_yads['PRIOR_DATE'].notnull()].copy()
        included_patients_yads = included_patients_yads[included_patients_yads['SUBSQ_DATE'].notnull()].copy()
        included_patients_yads[['PRIOR_DATE', 'SUBSQ_DATE']].applymap(lambda x: datetime.strptime(x, '%Y%m%d'))
        included_patients_yads = included_patients_yads.sort_values(by=['AnonymizedName', 'PRIOR_DATE'])
        included_patients_yads.to_csv(args.path_to_save_file + '4.YADS_with_included_patients.csv', na_rep='NULL', index=False, encoding='utf-8')
        return included_patients_yads
    
    
    def read_and_get_included_patients(self, yads_data):
        # Reading patiens file, changing the name, and applying the same format date to the date_of_death column
        patients_data = pd.read_csv(self.patients_file_to_read)
        patients_data.rename(columns={'avl_id' : 'PatientID'}, inplace = True)
        #patients_data['date_of_death'] = pd.to_datetime(patients_data['date_of_death'], format='%Y-%m-%d', errors='coerce')
        # Selecting date_of_death and date_last_check from the patients file with the same ID as yads_date preprocessed file
        date_of_death   = patients_data.set_index('PatientID')['date_of_death'].to_dict()
        date_last_check = patients_data.set_index('PatientID')['date_last_check'].to_dict()
        yads_data['DateOfDeath']     = yads_data['PatientID'].map(date_of_death)
        yads_data['DateOfLastCheck'] = yads_data['PatientID'].map(date_last_check)
        # Saving the dataframe which includes the date of death and the date of last check from the patients file
        yads_data.to_csv(self.path_to_save_file + '5.YADS_including_date_of_death_and_last_check.csv', na_rep='NULL', index=False, encoding='utf-8')
        return yads_data
    
    
    def get_days_of_survival(self, date_of_death, subsq_date, date_of_last_check):
        dates = []
        if isinstance(date_of_death, str):
            date_of_death = parse(date_of_death)
            date_of_death = date_of_death.strftime('%Y-%m-%d')
            dates = [datetime.strptime(str(date_of_death), '%Y-%m-%d').strftime('%Y-%m-%d'),
                     datetime.strptime(str(subsq_date), '%Y%m%d').strftime('%Y-%m-%d')]
        else:
            dates = [datetime.strptime(str(date_of_last_check), '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d"),
                     datetime.strptime(str(subsq_date), '%Y%m%d').strftime('%Y-%m-%d')]
        survival_days = ( datetime.strptime(dates[0], '%Y-%m-%d') - datetime.strptime(dates[1], '%Y-%m-%d')).days
        return survival_days
        
        
    def get_survival_from_patients(self, yads_data):
        # Survival
        yads_data.insert(len(yads_data.columns), 'DaysOfSurvival', 
                         yads_data.apply(lambda x: self.get_days_of_survival(x.DateOfDeath, 
                                                                          x.SUBSQ_DATE, x.DateOfLastCheck), axis=1))
        # Event
        yads_data.insert(len(yads_data.columns), 'Event', yads_data['DateOfDeath'].apply(lambda x: isinstance(x, str)))
        # y1Survival
        yads_data.insert(len(yads_data.columns), 'Y1Survival', yads_data['DaysOfSurvival'].apply(lambda x: int(x >= self.y1_survival_days)))
        # y2Survival
        yads_data.insert(len(yads_data.columns), 'Y2Survival', yads_data['DaysOfSurvival'].apply(lambda x: int(x >= self.y2_survival_days)))
        
        yads_data.to_csv(self.path_to_save_file + '6.YADS_survival_data.csv', na_rep='NULL', index=False, encoding='utf-8')
        return yads_data
    
    
    def format_date_of_death(self, date_of_death):
        
        if isinstance(date_of_death, str):
            date_of_death = parse(date_of_death)
            date_of_death = date_of_death.strftime('%Y-%m-%d')
            date_of_death = datetime.strptime(str(date_of_death), '%Y-%m-%d').strftime('%Y-%m-%d')
        return date_of_death
    
    
    def final_format_dates(self, yads_data):
        yads_data['PRIOR_DATE']      = yads_data['PRIOR_DATE'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%Y-%m-%d'))
        yads_data['SUBSQ_DATE']      = yads_data['SUBSQ_DATE'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%Y-%m-%d'))
        yads_data['DateOfLastCheck'] = yads_data['DateOfLastCheck'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d"))
        yads_data['DateOfDeath']     = yads_data['DateOfDeath'].apply(self.format_date_of_death)
        yads_data.to_csv(self.path_to_save_file + '7.YADS_survival_data_standardized.csv', na_rep='NULL', index=False, encoding='utf-8')


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
    
    preproc = Preprocessing(args)
    if args.task == 'preprocessing':
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
        
    elif args.task == 'generate_pairs':
        print(' Generating pairs file ....')
        preproc.get_pairs()
        print('End Generating pairs file! :)')
    
    else:
        print('Error! ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing of YADS and patient files to create pairs')
    
    # Variables to preprocess the NKI files: YADS file and patients file
    parser.add_argument('--task',               default='generate_pairs', type=str,  help='Taks to execute: preprocessing or generate_pairs')
    parser.add_argument('--YADS_file_to_read',  default='/projects/disentanglement_methods/files_nki/YADSRequestResult.csv', type=str, help='YADS file path')
    parser.add_argument('--patients_file_to_read', default='/projects/disentanglement_methods/files_nki/patients.csv', type=str, help='Patients file path')
    parser.add_argument('--min_days_inclusion', default=30,   type=int,   help='Minimum days between prior and subsquent CTs')
    parser.add_argument('--max_days_inclusion', default=120,  type=int,   help='Maximum days between prior and subsquent CTs')
    parser.add_argument('--y1_survival_days',   default=355,  type=int,   help='Number of days to assign survival')
    parser.add_argument('--y2_survival_days',   default=710,  type=int,   help='Number of days to assign survival')
    parser.add_argument('--path_to_save_file',  default='/projects/disentanglement_methods/files_nki/', type=str, help='Path to save the preprocessed file of YADS')
    parser.add_argument('--path_to_cts_in_yads',default='/data/groups/beets-tan/s.trebeschi/INFOa_dicoms/DICOM/', type=str, help='Path to add to the AnonymizedName in yalds to read the CTs')
    
    # Variables for generating the pairs
    parser.add_argument('--name_file_to_read',           default='/projects/disentanglement_methods/files_nki/7.YADS_survival_data_standardized.csv', type=str, help='File which contains initial pairs')
    parser.add_argument('--name_pairs_file',             default='/projects/disentanglement_methods/files_nki/infoA/pairs.csv', type=str, help='Path to save file with all the pairs for infoA')
    
    args = parser.parse_args()
    
    main(args)