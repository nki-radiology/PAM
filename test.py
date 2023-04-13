import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data               import DataLoader
from utils.utils_torch              import weights_init
from sklearn.model_selection        import train_test_split
from pathlib                        import Path


from RegistrationDataset            import RegistrationDataSet
from networks.PAMNetwork            import PAMNetwork
from networks.DiscriminatorNetwork  import DiscriminatorNetwork
from metrics.LossPam                import Energy_Loss, Cross_Correlation_Loss

from config import PARAMS

RANDOM_SEED = 42


def read_train_data():
    path_input= PARAMS.train_folder
    path      = Path(path_input)
    filenames = list(path.glob('*.nrrd'))
    data_index= []

    for f in filenames:
        data_index.append(int(str(f).split('/')[7].split('_')[0])) 

    train_data = list(zip(data_index, filenames))
    train_data = pd.DataFrame(train_data, columns=['tcia_idx', 'dicom_path'])

    return train_data


def cuda_seeds():
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark    = False


def init_loss_functions():
    discriminator_loss = nn.BCELoss()  
    l2_loss = nn.MSELoss()  
    nn_loss = Cross_Correlation_Loss().pearson_correlation
    penalty = Energy_Loss().energy_loss
    return discriminator_loss, l2_loss, nn_loss, penalty


def load_dataloader():
    filenames   = read_train_data()

    _, inputs_test = train_test_split(
        filenames, random_state=RANDOM_SEED, train_size=0.8, shuffle=True
    )

    test_dataset    = RegistrationDataSet(
        path_dataset = inputs_test,
        input_shape  = (300, 192, 192, 1),
        transform    = None
    )
    
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return test_dataloader


def load_model_weights():
    # Network definition
    pam_net     = PAMNetwork(PARAMS.img_dim, PARAMS.filters)
    device      = torch.device('cuda:0')
    pam_net.to(device)

    # Loading the model weights
    pam_chkpt = os.path.join(PARAMS.project_folder, 'PAMModel.pth')
    pam_net.load_state_dict(torch.load(pam_chkpt))

    return pam_net, device


def measure_disentaglement(pam_network, fixed, moving, effect=1.):
    # get embedding of the deformation
    z, (_, _) = pam_network.get_features(fixed, moving)
    # modify entry 
    ix = np.random.randint(z.shape[1])
    z[:, ix] += effect
    # generate and apply deformation to moving
    _, _, _, moving_ = pam_network.generate(z, moving)
    # moving_ should match fixed (z=0), except for the feature modified
    z, (_, _) = pam_network.get_features(fixed, moving_)
    ix_pred = torch.argmax(torch.abs(z))
    return ix == ix_pred


def test(pam_network, test_dataloader, device):

    _, _, cc_loss, penalty = init_loss_functions()

    pam_network.eval()
    results = []

    print('starting...')

    for i, (x_1, x_2) in enumerate(test_dataloader):

        print('sample', str(i), end='\t')
                    
        fixed  = x_1.to(device)
        moving = x_2.to(device)

        t_0, w_0, t_1, w_1 = pam_network(fixed, moving)
        z, (z_fixed, z_moving) = pam_network.get_features(fixed, moving)

        print('registered', end='\t')

        registration_affine_loss = cc_loss(fixed, w_0)
        penalty_affine_loss      = penalty(t_0) 

        print('affine:', str(registration_affine_loss.cpu().detach().numpy().squeeze()), end='\t')

        registration_deform_loss = cc_loss(fixed, w_1)
        penalty_deform_loss = penalty(t_1)

        print('elastic:', str(registration_deform_loss.cpu().detach().numpy().squeeze()), end='\t')

        matched = measure_disentaglement(pam_network, fixed, moving, effect=torch.std(z))

        print('disentangl:', str(matched.cpu().detach().numpy().squeeze()))

        results.append({
            'reg_aff'   : registration_affine_loss,
            'pen_aff'   : penalty_affine_loss,
            'reg_def'   : registration_deform_loss,
            'pen_def'   : penalty_deform_loss,
            'dis'       : matched ,
            'z'         : z,
            'z_fixed'   : z_fixed,
            'z_moving'  : z_moving
        })

        pd.DataFrame(results).to_csv('test.csv')

    print('done.')

cuda_seeds()
pam_network, device = load_model_weights()
test_dataloader     = load_dataloader()

with torch.no_grad():
    test(pam_network, test_dataloader, device)



