import pandas as pd

def assign(entry):
        if entry.PatientID % 2 == 0:
            fold = 'train'
        else:
            fold = 'test'
        return fold
        
        
def add_valid_fold(filename_to_read, filename_to_save):
    df = pd.read_csv(filename_to_read)
    # Create a boolean mask to select the rows where fold is 'train' and id_patient is divisible by 10
    mask = (df['fold'] == 'train') & (df['PatientID'] % 10 == 0)
    # Update the fold column for the selected rows
    df.loc[mask, 'fold'] = 'valid'
    df.to_csv(filename_to_save, na_rep='NULL', index=False, encoding='utf-8')
    print('Saving data including train, valid, and test as folds!')


def get_data_file(filename_to_read, filename_to_save):
    infoA = pd.read_csv(filename_to_read, index_col=0)
    print('Initial CT scans number: ', len(infoA))
    
    # Adding train and test fold
    infoA.insert(len(infoA.columns), 'fold', infoA.apply(assign, axis=1))
    infoA.to_csv(filename_to_save, na_rep='NULL', index=False, encoding='utf-8')
    print('Saving data including train and test as folds!')
        

get_data_file('/projects/disentanglement_methods/files_nki/infoA/abdomen/abdomen_pairs_only_succeed_nrrd.csv', '/projects/disentanglement_methods/files_nki/survival_net_files/abdomen_pairs_train_test.csv')
add_valid_fold('/projects/disentanglement_methods/files_nki/survival_net_files/abdomen_pairs_train_test.csv', '/projects/disentanglement_methods/files_nki/survival_net_files/abdomen_pairs_train_valid_test.csv')