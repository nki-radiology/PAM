import pandas as pd
from  pathlib import Path

class SavePairs():
    def __init__(self, filename_to_read: str, anatomy_to_read: str, filename_including_non_succeed_nrrd: str, filename_only_succeed_nrrd:str):
        self.filename_to_read = filename_to_read
        self.anatomy_to_read  = anatomy_to_read
        self.filename_including_non_succeed_nrrd = filename_including_non_succeed_nrrd
        self.filename_only_succeed_nrrd          = filename_only_succeed_nrrd
        
        self.path_to_change   = '/data/groups/beets-tan/s.trebeschi/INFOa_dicoms/DICOM/'
        self.new_path_name    = '/data/groups/beets-tan/l.estacio/infoA/' + self.anatomy_to_read + '/'


    def assign_prior_nrrd_path(self, entry):
        prior_path = entry.PRIOR_PATH.replace(self.path_to_change, self.new_path_name)
        prior_path = prior_path + '/' + prior_path.split('/')[9] + '.nrrd'
        return prior_path

    def assign_subsq_nrrd_path(self, entry):
        subsq_path = entry.SUBSQ_PATH.replace(self.path_to_change, self.new_path_name)
        subsq_path = subsq_path + '/' + subsq_path.split('/')[9] + '.nrrd'
        return subsq_path

    def assign_prior_nrrd_path_succeed(self, entry):
        succeed = Path(entry.PRIOR_PATH_NRRD).is_file()
        return succeed
    
    def assign_subsq_nrrd_path_succeed(self, entry):
        succeed = Path(entry.SUBSQ_PATH_NRRD).is_file()
        return succeed

    def get_succed_pairs(self):
        # Reading the original csv file
        data  = pd.read_csv(self.filename_to_read)
        print(' Initial data length: ', len(data))
        
        # Adding the NRRD path
        data.insert(len(data.columns), 'PRIOR_PATH_NRRD', data.apply(self.assign_prior_nrrd_path, axis=1))
        data.insert(len(data.columns), 'SUBSQ_PATH_NRRD', data.apply(self.assign_subsq_nrrd_path, axis=1))
        
        # Adding succeed NRRD paths
        data.insert(len(data.columns), 'PRIOR_PATH_NRRD_SUCCEED', data.apply(self.assign_prior_nrrd_path_succeed, axis=1))
        data.insert(len(data.columns), 'SUBSQ_PATH_NRRD_SUCCEED', data.apply(self.assign_subsq_nrrd_path_succeed, axis=1))
        data.to_csv(self.filename_including_non_succeed_nrrd, na_rep='NULL', index=False, encoding='utf-8')
    
    def keep_only_succeed_pairs(self):
        data  = pd.read_csv(self.filename_including_non_succeed_nrrd)
            
        data = data.loc[data["PRIOR_PATH_NRRD_SUCCEED"] & data["SUBSQ_PATH_NRRD_SUCCEED"] ]
        data.to_csv(self.filename_only_succeed_nrrd, na_rep='NULL', index=True, encoding='utf-8')
        
        
save = SavePairs('/projects/disentanglement_methods/files_nki/infoA/pairs.csv', 'abdomen', '/projects/disentanglement_methods/files_nki/infoA/abdomen_pairs_including_non_succeed_nrrd.csv', '/projects/disentanglement_methods/files_nki/infoA/abdomen_pairs_only_succeed_nrrd.csv')
save.get_succed_pairs()
save.keep_only_succeed_pairs()