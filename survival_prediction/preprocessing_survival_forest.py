import pandas as pd
from datetime import datetime
from dateutil.parser import parse

def format_date(date_treatment):
        
        if isinstance(date_treatment, str):
            date_treatment = parse(date_treatment)
            date_treatment = date_treatment.strftime('%Y-%m-%d')
            date_treatment = datetime.strptime(str(date_treatment), '%Y-%m-%d').strftime('%Y-%m-%d')
        return date_treatment


def get_treatment_dates_from_patients(surv_file, patients_file, path_to_save_file):
        # Reading patiens file, changing the name, and applying the same format date to the date_of_death column
        patients_data = pd.read_csv(patients_file)
        surv_data     = pd.read_csv(surv_file)
        patients_data.rename(columns={'avl_id' : 'PatientID'}, inplace = True)
       
        # Selecting date_of_death and date_last_check from the patients file with the same ID as yads_date preprocessed file
        date_start_first_immunotherapy = patients_data.set_index('PatientID')['date_start_first_immunotherapy'].to_dict()
        date_start_last_immunotherapy  = patients_data.set_index('PatientID')['date_start_last_immunotherapy'].to_dict()
        surv_data['DateStartFirstImmunotherapy'] = surv_data['PatientID'].map(date_start_first_immunotherapy)
        surv_data['DateStartLastImmunotherapy']  = surv_data['PatientID'].map(date_start_last_immunotherapy)
        
        # Format of dates
        surv_data['DateStartFirstImmunotherapy'] = surv_data['DateStartFirstImmunotherapy'].apply(format_date)
        surv_data['DateStartLastImmunotherapy']  = surv_data['DateStartLastImmunotherapy'].apply(format_date)
        
        # Saving the dataframe which includes the date of death and the date of last check from the patients file
        surv_data.to_csv(path_to_save_file + '1.features_pam_including_dates_of_treatment.csv', na_rep='NULL', index=False, encoding='utf-8')
        print('File Saved! ')

def get_days_between_prior_start_treatment(prior_date, date_of_start_treatment):
        dates = []
        dates = [datetime.strptime(str(prior_date), '%Y-%m-%d').strftime("%Y-%m-%d"),
                 datetime.strptime(str(date_of_start_treatment), '%Y-%m-%d').strftime('%Y-%m-%d')]
        survival_days = ( datetime.strptime(dates[0], '%Y-%m-%d') - datetime.strptime(dates[1], '%Y-%m-%d')).days
        return survival_days


def get_difference_between_dates_of_treatment(surv_file_to_read, path_to_save_file):
        surv_data = pd.read_csv(surv_file_to_read)
        # Assigning the proper path of scans, number of scans, and saving it as csv file.
        surv_data.insert(len(surv_data.columns), 'DifferenceInDaysBetweenPriorAndStartTreatment', 
                         surv_data.apply(lambda x: get_days_between_prior_start_treatment(x.PRIOR_DATE, 
                                                                          x.DateStartFirstImmunotherapy), axis=1))
        surv_data.to_csv(path_to_save_file + '2.features_pam_including_difference_between_dates_of_treatment.csv', na_rep='NULL', index=False, encoding='utf-8')
        print('File Saved! ')

def main():
        surv_file         = '/projects/disentanglement_methods/files_nki/infoA/abdomen/features/features_pam_abdomen_mean.csv'
        patients_file     = '/projects/disentanglement_methods/files_nki/preprocessing_pairs/patients.csv'
        path_to_save_file = '/projects/disentanglement_methods/files_nki/infoA/abdomen/features/'
        get_treatment_dates_from_patients(surv_file, patients_file, path_to_save_file)
        get_difference_between_dates_of_treatment(path_to_save_file + '1.features_pam_including_dates_of_treatment.csv', path_to_save_file)
        

if __name__ == '__main__':
    main()
    