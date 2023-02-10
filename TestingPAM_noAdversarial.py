# Set up the CUDA device
import os

# Set up the argparse variables by default
import sys
sys.argv = ['']

# General lybraries
import torch
import pandas                     as     pd
import matplotlib.pyplot          as     plt
from   torch.utils.data           import DataLoader
from   pathlib                    import Path

# Helper Functions
from general_config               import arg_pam_test_fts, deformation
from utils.utils                  import create_directory

# Model Classes
from networks.PAMNetwork           import PAMNetwork
#from networks.DiscriminatorNetwork import DiscriminatorNetwork
from RegistrationDataset           import RegistrationDataSet
from metrics.LossPam               import *

def read_test_data():
    path_input= arg_pam_test_fts.test_folder
    path      = Path(path_input)
    filenames = list(path.glob('*.nrrd'))
    data_index= []

    for f in filenames:
           data_index.append(int(str(f).split('/')[6].split('_')[0]))
        

    train_data = list(zip(data_index, filenames))
    train_data = pd.DataFrame(train_data, columns=['tcia_idx', 'dicom_path'])

    return train_data

#read_test_data()

def model_init():
    # Network definition
    pam_net     = PAMNetwork()
    
    # GPU computation
    device      = torch.device('cuda:0')
    pam_net.to(device)
    
    # Loading the model weights
    pam_chkpt = arg_pam_test_fts.pam_checkpoint
    pam_net.load_state_dict(torch.load(pam_chkpt))

    return pam_net, device

def load_data():
    
    # Dataset paths
    file_names      = read_test_data()
    print(file_names)
    
    # Testing dataset
    test_dataset    = RegistrationDataSet(path_dataset = file_names,
                                          input_shape  = (300, 192, 192, 1),
                                          transform    = None
                                         )
    
    # Testing dataloader
    test_dataloader = DataLoader(dataset    = test_dataset, 
                                 batch_size = 1, 
                                 shuffle    = True
                                )
    
    return test_dataloader

def mkdir_results_folder():
    
    folder_name = arg_pam_test_fts.results_folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
def visualize_slices(fixed, moving, w_0, t_0, w_1, t_1, i):
    fixed_copy  = torch.squeeze(fixed.cpu(),  0)
    moving_copy = torch.squeeze(moving.cpu(), 0)
    w_0_copy    = torch.squeeze(w_0.cpu(), 0)
    t_0_copy    = torch.squeeze(t_0.cpu(), 0)
    w_1_copy    = torch.squeeze(w_1.cpu(), 0)
    t_1_copy    = torch.squeeze(t_1.cpu(), 0)
    f, axarr    = plt.subplots(3,2)
    axarr[0,0].imshow(fixed_copy [0, :, :, 140])
    axarr[0,0].set_title('Fixed')
    axarr[0,1].imshow(moving_copy[0, :, :, 140])
    axarr[0,1].set_title('Moving')
    axarr[1,0].imshow(w_0_copy   [0, :, :, 140])
    axarr[1,0].set_title('Affine')
    axarr[1,1].imshow(t_0_copy  )
    axarr[1,1].set_title('Affine Matrix')
    axarr[2,0].imshow(w_1_copy   [0, :, :, 140])
    axarr[2,0].set_title('Deformation')
    axarr[2,1].imshow(t_1_copy   [0, :, :, 140])
    axarr[2,1].set_title('Deformation field')

    plt.savefig(os.path.join(arg_pam_test_fts.results_folder, '{}.png').format(i))
    
def start_test(pam_net, device, test_dataloader):
    
    # Evaluation mode
    pam_net.eval()
    
    # Loss for testing
    loss_pam = 0    

    alpha_value  = 0.01
    beta_value   = 0.1
    
    for i, (x_1, x_2) in enumerate(test_dataloader):
        
        # Reading the images and sending to the device (GPU by default)
        fixed = x_1.to(device)
        moving= x_2.to(device)        
        
        with torch.no_grad():
            # Forward pass through the PAM model
            t_0, w_0, t_1, w_1 = pam_net(fixed, moving)            
            
            # Compute the affine loss
            sim_af_loss, reg_af_loss = total_loss(fixed, w_0, w_0)
            total_affine             = sim_af_loss + alpha_value * reg_af_loss
            
            # Compute the deformation/elastic loss
            sim_df_loss, reg_df_loss = total_loss(fixed, w_1, t_1)
            total_elastic            = sim_df_loss + beta_value * reg_df_loss
            
            # PAM loss
            loss          = total_affine + total_elastic
            loss_pam     += loss.item()
            print(f'PAM Loss image {i}: {loss.item()}')
            
            visualize_slices(fixed, moving, w_0, t_0, w_1, t_1, i)            
                    
    # Mean loss
    loss_pam /= len(test_dataloader)
    print("Testing : loss_PAM = {:.6f},".format(loss_pam))

#--------------------------#
""" STARTING THE TESTING """
pam_net, device = model_init()
test_dataloader = load_data()
mkdir_results_folder()
start_test(pam_net, device, test_dataloader)