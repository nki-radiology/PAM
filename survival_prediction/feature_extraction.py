import os
import sys
import torch
import pandas  as     pd
from   tqdm    import tqdm
from   pydicom import dcmread
sys.path.append('/projects/disentanglement_methods/PAM/')
from networks.PAMNetFeatures       import PAMNetwork
from networks.DiscriminatorNetwork import DiscriminatorNetwork
from libs.frida.io                 import ImageLoader, ReadVolume
from libs.frida.transforms         import ZeroOneScaling, ToNumpyArray


class PamModel:

    def __init__(self, pam_checkpoint, dis_checkpoint):
        self.pam_net = PAMNetwork()
        self.dis_net = DiscriminatorNetwork()
        self.pam_ckp = pam_checkpoint
        self.dis_ckp = dis_checkpoint
        self.device  = "cuda:0" # change

    def assign_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pam_net.to(self.device)
        self.dis_net.to(self.device)

    def load_pam_weights(self):
        self.pam_net.load_state_dict(torch.load(self.pam_ckp))
        self.dis_net.load_state_dict(torch.load(self.dis_ckp))

    def assign_eval_mode(self):
        self.pam_net.eval()
        self.dis_net.eval()

    def get_features(self, fx, mv):
        fixed  = fx.to(self.device)
        moving = mv.to(self.device)
        transform_0, warped_0, e4, e5, d4, transform_1, warped_1 = self.pam_net(fixed, moving)
        return e4, e5, d4


class PamFeatures():

    def __init__(self, pam_checkpoint, dis_checkpoint):
        self.pam             = PamModel(pam_checkpoint, dis_checkpoint)
        self.patients        = []
        self.tag_bl_list     = []
        self.tag_fu_list     = []
        self.pam_ls_mean_list= []
        self.pam_ls_max_list = []
        self.pam_ls_quan_list= []
        
        # No readed variables
        self.exluded_patients = []

        # Loading weights of the model
        self.pam.assign_device()
        self.pam.load_pam_weights()
        self.pam.assign_eval_mode()


    def load_dicom_tagssafely(self, path, scan, prefix = ''):
        # wraps metatags loading around a try-catch
        # attach a prefix to the fields if needed
        result = None

        try:
            dcm = os.path.join(path, os.listdir(path)[0])
            ds  = dcmread(dcm)

            tags = (
                0x00080020, # Study Date
                0x00081030, # Study Description
                0x00180060, # KVP
                0x00280030, # Pixel Spacing
                0x00180050, # Slice Thickness
                0x00180088, # Spacing Between Slices
                0x00189306, # Single Collimation Width
                0x00189307, # Total Collimation Width
                0x00181151, # X-Ray Tube Current
                0x00181210, # Convolution Kernel
                0x00181150, # Exposure Time
                0x00189311  # Spiral Pitch Factor
            )
            result = dict()
            for t in tags:
                try:
                    descr = ds[t].description()
                    descr = descr.replace(' ', '').replace('-', '')
                    descr = prefix + descr.lower() + scan
                    result.update({descr: ds[t].value})
                except:
                    pass
            print(' - [OK] DICOM tags loaded correctly ')

        except:
            print(' - [failed] while loading of the DICOM tags. ' )
        return result


    def zero_at_edges(self, fx):
        #IMAGE HAS VALUE OF ZERO AT EDGES
        fx[:,  :,  0]  = 0
        fx[:,  :, -1]  = 0
        fx[:,  0,  :]  = 0
        fx[:, -1,  :]  = 0
        fx[0,  :,  :]  = 0
        fx[-1, :,  :]  = 0
        return fx
    

    def pam_features(self, fx, mv):
        fx = fx.transpose(3, 1, 2, 0)
        fx = torch.from_numpy(fx).type(torch.float32)
        fx = fx[None, :]

        mv = mv.transpose(3, 1, 2, 0)
        mv = torch.from_numpy(mv).type(torch.float32)
        mv = mv[None, :]
        e4, e5, d4 = self.pam.get_features(fx, mv)

        # Latent space
        _, dim1, dim2, dim3, dim4 = e5.size()
        ls_layer                  = torch.reshape(e5, (dim1, dim2*dim3*dim4))

        # Features based on the mean values
        ls_layer_mean             = torch.mean(ls_layer, 1)
        ls_layer_mean_non_zero    = torch.count_nonzero(ls_layer_mean) / dim1
        std                       = torch.std(ls_layer_mean, unbiased=False)
        mean                      = torch.mean(ls_layer_mean)
        ls_layer_mean_lambda_mo   = (std ** 2 + mean ** 2) / (mean + 1e-10) - 1
        ls_layer_mean_pi_mo       = (std ** 2 - mean) / (std ** 2 + mean ** 2 - mean + 1e-10)

        # Features based on the maximum values
        ls_layer_max, _           = torch.max(ls_layer, 1)
        ls_layer_max_non_zero     = torch.count_nonzero(ls_layer_max) / dim1
        std                       = torch.std(ls_layer_max, unbiased=False)
        mean                      = torch.mean(ls_layer_max)
        ls_layer_max_lambda_mo    = (std ** 2 + mean ** 2) / (mean + 1e-10) - 1
        ls_layer_max_pi_mo        = (std ** 2 - mean) / (std ** 2 + mean ** 2 - mean + 1e-10)

        # Features based on the percentile values
        ls_layer_quantile          = torch.quantile(ls_layer, 0.975, 1)
        ls_layer_quantile_non_zero = torch.count_nonzero(ls_layer_quantile) / dim1
        std                        = torch.std(ls_layer_quantile, unbiased=False)
        mean                       = torch.mean(ls_layer_quantile)
        ls_layer_quantile_lambda_mo= (std ** 2 + mean ** 2) / (mean + 1e-10) - 1
        ls_layer_quantile_pi_mo    = (std ** 2 - mean) / (std ** 2 + mean ** 2 - mean + 1e-10)

        # Save feature map 
        result_mean     = dict()
        result_max      = dict()
        result_quantile = dict()
        
        # Save additional values
        for i, f in enumerate(ls_layer_mean):
            result_mean.update({'feature_' + str(i): f.item()})
        result_mean.update({'non_zero_percentage': ls_layer_mean_non_zero.item()})
        result_mean.update({'lambda_mo': ls_layer_mean_lambda_mo.item()})
        result_mean.update({'pi_mo': ls_layer_mean_pi_mo.item()})
        
        for i, f in enumerate(ls_layer_max):
            result_max.update({'feature_' + str(i): f.item()})
        result_max.update({'non_zero_percentage': ls_layer_max_non_zero.item()})
        result_max.update({'lambda_mo': ls_layer_max_lambda_mo.item()})
        result_max.update({'pi_mo': ls_layer_max_pi_mo.item()})
        
        for i, f in enumerate(ls_layer_quantile):
            result_quantile.update({'feature_' + str(i): f.item()})
        result_quantile.update({'non_zero_percentage': ls_layer_quantile_non_zero.item()})
        result_quantile.update({'lambda_mo': ls_layer_quantile_lambda_mo.item()})
        result_quantile.update({'pi_mo': ls_layer_quantile_pi_mo.item()})
        
        return result_mean, result_max, result_quantile

    

    def get_features(self, filename: str, path_to_save_files: str, name_to_save_xlsx: str):

        # Reading the original csv file
        data = pd.read_csv(filename)
        
        loader = ImageLoader(
            ReadVolume(),
            ZeroOneScaling(),
            ToNumpyArray(add_singleton_dim=True)
        )
        
        with tqdm(total=len(data)) as pbar: 
            for idx in range(len(data)):

                # Baselines: DICOM and NRRD
                baseline_dicom      = data['PRIOR_PATH'].iloc[idx]
                baseline_nrrd       = data['PRIOR_PATH_NRRD'].iloc[idx]

                # Followup: DICOM and NRRD
                follow_up_dicom     = data['SUBSQ_PATH'].iloc[idx]
                follow_up_nrrd      = data['SUBSQ_PATH_NRRD'].iloc[idx]

                # Get Tags from the baseline and followup (DICOM)
                baseline_dicom_tag  = self.load_dicom_tagssafely(baseline_dicom, '_PRIOR')
                follow_up_dicom_tag = self.load_dicom_tagssafely(follow_up_dicom, '_SUBSQ')


                # Get Pam features (NRRD)
                try:
                    bl_img  = self.zero_at_edges(loader(baseline_nrrd))
                    fu_img  = self.zero_at_edges(loader(follow_up_nrrd))
                    pam_ls_mean, pam_ls_max, pam_ls_quantile  = self.pam_features (bl_img, fu_img)
                    
                    print('Baseline nrrd: ', baseline_nrrd)
                    print('Follow-up NRRd: ', follow_up_nrrd)

                    # Save dictionaries
                    
                    self.patients.append({'AnonymizedName': data['AnonymizedName'].iloc[idx], 'PatientID': data['PatientID'].iloc[idx], 
                                          'YADSId': data['YADSId'].iloc[idx], 'AnonymizedPatientID': data['AnonymizedPatientID'].iloc[idx],
                                          'Included': data['Included'].iloc[idx], 'PRIOR_DATE': data['PRIOR_DATE'].iloc[idx],
                                          'PRIOR_PATH': baseline_dicom,'SUBSQ_DATE': data['SUBSQ_DATE'].iloc[idx], 'SUBSQ_PATH': follow_up_dicom,
                                          'DifferenceInDaysBetweenScans':data['DifferenceInDaysBetweenScans'].iloc[idx], 'DateOfDeath':data['DateOfDeath'].iloc[idx],
                                          'DateOfLastCheck': data['DateOfLastCheck'].iloc[idx], 'DaysOfSurvival': data['DaysOfSurvival'].iloc[idx],
                                          'Event': data['Event'].iloc[idx], 'Y1Survival': data['Y1Survival'].iloc[idx], 'Y2Survival': data['Y2Survival'].iloc[idx],
                                          'PRIOR_PATH_NRRD': data['PRIOR_PATH_NRRD'].iloc[idx], 'SUBSQ_PATH_NRRD': data['SUBSQ_PATH_NRRD'].iloc[idx]
                                        })
                    # DICOM tags
                    if not baseline_dicom_tag: 
                            baseline_dicom_tag = {'studydate': ' ', 'studydescription': ' ', 'kvp': ' ', 'pixelspacing': ' ', 
                            'slicethickness': ' ', 'xraytubecurrent': ' ', 'convolutionkernel': ' ', 'exposuretime': ' ', 'spiralpitchfactor': ' '}
                    self.tag_bl_list.append(baseline_dicom_tag)

                    if not follow_up_dicom_tag: 
                            follow_up_dicom_tag = {'studydate': ' ', 'studydescription': ' ', 'kvp': ' ', 'pixelspacing': ' ', 
                            'slicethickness': ' ', 'xraytubecurrent': ' ', 'convolutionkernel': ' ', 'exposuretime': ' ', 'spiralpitchfactor': ' '}
                    self.tag_fu_list.append(follow_up_dicom_tag)
                    
                    # PAM features
                    self.pam_ls_mean_list.append(pam_ls_mean)
                    self.pam_ls_max_list.append(pam_ls_max)
                    self.pam_ls_quan_list.append(pam_ls_quantile)

                    # Saving csv
                    patient_after    = pd.DataFrame.from_dict(self.patients)
                    tag_bl_all_after = pd.DataFrame.from_dict(self.tag_bl_list)
                    tag_fu_all_after = pd.DataFrame.from_dict(self.tag_fu_list)
                    pam_ls_all_mean  = pd.DataFrame.from_dict(self.pam_ls_mean_list)
                    pam_ls_all_max   = pd.DataFrame.from_dict(self.pam_ls_max_list)
                    pam_ls_all_quan  = pd.DataFrame.from_dict(self.pam_ls_quan_list)

                    new_df_mean      = pd.concat([patient_after, tag_bl_all_after, tag_fu_all_after, pam_ls_all_mean], axis=1, ignore_index=False, sort=False)
                    new_df_max       = pd.concat([patient_after, tag_bl_all_after, tag_fu_all_after, pam_ls_all_max], axis=1, ignore_index=False, sort=False)
                    new_df_quan      = pd.concat([patient_after, tag_bl_all_after, tag_fu_all_after, pam_ls_all_quan], axis=1, ignore_index=False, sort=False)
                    
                    new_df_mean.to_csv(path_to_save_files + name_to_save_xlsx + '_mean.csv',       na_rep='NULL', index=True, encoding='utf-8')
                    new_df_max.to_csv(path_to_save_files  + name_to_save_xlsx + '_max.csv',        na_rep='NULL', index=True, encoding='utf-8')
                    new_df_quan.to_csv(path_to_save_files + name_to_save_xlsx + '_percentile.csv', na_rep='NULL', index=True, encoding='utf-8')
                
                except:
                    print(' - [failed] while loading of the NRRD images. ' )
                    
                    self.exluded_patients.append({'AnonymizedName': data['AnonymizedName'].iloc[idx], 'PatientID': data['PatientID'].iloc[idx], 
                                          'YADSId': data['YADSId'].iloc[idx], 'AnonymizedPatientID': data['AnonymizedPatientID'].iloc[idx],
                                          'Included': data['Included'].iloc[idx], 'PRIOR_DATE': data['PRIOR_DATE'].iloc[idx],
                                          'PRIOR_PATH': baseline_dicom,'SUBSQ_DATE': data['SUBSQ_DATE'].iloc[idx], 'SUBSQ_PATH': follow_up_dicom,
                                          'DifferenceInDaysBetweenScans':data['DifferenceInDaysBetweenScans'].iloc[idx], 'DateOfDeath':data['DateOfDeath'].iloc[idx],
                                          'DateOfLastCheck': data['DateOfLastCheck'].iloc[idx], 'DaysOfSurvival': data['DaysOfSurvival'].iloc[idx],
                                          'Event': data['Event'].iloc[idx], 'Y1Survival': data['Y1Survival'].iloc[idx], 'Y2Survival': data['Y2Survival'].iloc[idx],
                                          'PRIOR_PATH_NRRD': data['PRIOR_PATH_NRRD'].iloc[idx], 'SUBSQ_PATH_NRRD': data['SUBSQ_PATH_NRRD'].iloc[idx]
                                        })
                    
                    patient_nr    = pd.DataFrame.from_dict(self.exluded_patients)
                    #new_df_nr = pd.concat([patient_nr, prior_date_nr, prior_path_nr, subsq_date_nr, subsq_path_nr], axis=1, ignore_index=False, sort=False)
                    patient_nr.to_csv(path_to_save_files + name_to_save_xlsx + '_unprocessed_images.csv', na_rep='NULL', index=True, encoding='utf-8')
                pbar.update(1)

    
if __name__ == "__main__":
    
    from   argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument('--thorax_raw_file_path',      type=str, default='/projects/disentanglement_methods/files_nki/infoA/thorax/thorax_pairs_only_succeed_nrrd.csv')
    parser.add_argument('--thorax_pam_checkpoint',     type=str, default='/projects/disentanglement_methods/checkpoints/PAM/thorax/PAMModel_60.pth')
    parser.add_argument('--thorax_dis_checkpoint',     type=str, default='/projects/disentanglement_methods/checkpoints/PAM/thorax/DisModel_60.pth')
    parser.add_argument('--thorax_path_to_save_file',  type=str, default='/projects/disentanglement_methods/files_nki/infoA/thorax/')
    parser.add_argument('--thorax_name_to_save_xlsx',  type=str, default='features_pam_thorax')
    
    parser.add_argument('--abdomen_raw_file_path',     type=str, default='/projects/disentanglement_methods/files_nki/infoA/abdomen/abdomen_pairs_only_succeed_nrrd.csv')
    parser.add_argument('--abdomen_pam_checkpoint',    type=str, default='/projects/disentanglement_methods/checkpoints/PAM/abdomen/PAMModel_60.pth')
    parser.add_argument('--abdomen_dis_checkpoint',    type=str, default='/projects/disentanglement_methods/checkpoints/PAM/abdomen/DisModel_60.pth')
    parser.add_argument('--abdomen_path_to_save_file', type=str, default='/projects/disentanglement_methods/files_nki/infoA/abdomen/')
    parser.add_argument('--abdomen_name_to_save_xlsx', type=str, default='features_pam_abdomen')
    
    parser.add_argument('--region_to_get_features',    type=str, default='Thorax')
    args = parser.parse_args()

     
    if args.region_to_get_features =='Thorax':
        print(' ------------------------------- [THORAX] Features! ------------------------------- ' )
        pam_features = PamFeatures(args.thorax_pam_checkpoint, args.thorax_dis_checkpoint)
        pam_features.get_features(args.thorax_raw_file_path, args.thorax_path_to_save_file, args.thorax_name_to_save_xlsx)
        
    else:
        print(' ------------------------------- [ABDOMEN] Features! ------------------------------- ' )
        pam_features = PamFeatures(args.abdomen_pam_checkpoint, args.abdomen_dis_checkpoint)
        pam_features.get_features(args.abdomen_raw_file_path, args.abdomen_path_to_save_file, args.abdomen_name_to_save_xlsx)
    
    name_file = 'commandline_args_' + args.region_to_get_features + '.txt'
    with open(name_file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
