import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from pydicom                import dcmread

from networks.PAMNetwork    import RegistrationNetwork, RegistrationStudentNetwork
from metrics.LossPam        import Energy_Loss, Cross_Correlation_Loss

from libs.frida.io          import ImageLoader, ReadVolume
from libs.frida.transforms  import TransformFromNumpyFunction, ZeroOneScaling, ToNumpyArray


from config import PARAMS

RANDOM_SEED = 42

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
    # registration loss
    correlation     = Cross_Correlation_Loss().pearson_correlation
    energy          = Energy_Loss().energy_loss

    # adversatial loss
    binary_entropy  = nn.BCELoss()
    mse_distance    = nn.MSELoss()

    # latent loss
    l2_norm     = lambda x:torch.norm(x, p=2)
    l1_norm     = lambda x:torch.norm(x, p=1)

    return (correlation, energy), (binary_entropy, mse_distance), (l2_norm, l1_norm)   


def load_trained_models():
    # Network definition
    reg_net     = RegistrationNetwork(PARAMS.img_dim, PARAMS.filters)
    std_net     = RegistrationStudentNetwork(PARAMS.img_dim, PARAMS.filters, PARAMS.latent_dim)
    
    device      = torch.device('cuda:0')

    def load_model(model, name):
        model.to(device)
        path = os.path.join(PARAMS.project_folder, name + '.pth')
        model.load_state_dict(torch.load(path))

    load_model(reg_net, 'RegModel')
    load_model(std_net, 'StdModel')

    return reg_net, std_net, device


def load_dicom_tagssafely(path, prefix = ''):
        # wraps metatags loading around a try-catch
        # attach a prefix to the fields if needed
        result = {}

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
                    descr = prefix + descr.lower()
                    result.update({descr: ds[t].value})
                except:
                    pass
        except:
            print(' - [failed] while loading of the DICOM tags. ' )

        return result


def array_to_dict(array, name):
    result = {}
    for i, value in enumerate(array):
        result[f'{name}_{i}'] = value
    return result
    

def zero_at_edges(im):
    im[:,  :,  0]  = 0
    im[:,  :, -1]  = 0
    im[:,  0,  :]  = 0
    im[:, -1,  :]  = 0
    im[0,  :,  :]  = 0
    im[-1, :,  :]  = 0
    return im


def np2torch(arr):
    arr = arr.transpose(0, 1, 3, 4, 2)
    arr = torch.from_numpy(arr).type(torch.float32)
    return arr


def test(registration_network, student_network, dataset, device):

    dataset = pd.read_csv(dataset, index_col=0)
    print('loaded dataset', str(dataset.shape[0]), 'rows')

    loader = ImageLoader(
        ReadVolume(),
        ZeroOneScaling(),
        TransformFromNumpyFunction(zero_at_edges),
        ToNumpyArray(add_batch_dim=True, add_singleton_dim=True, channel_second=True)
    )

    (correlation, energy), (_, mse_distance), (_, _) = init_loss_functions()
    registration_network.eval()
    student_network.eval()

    result = []

    for i, row in dataset.iterrows():

        print('processing row', i, 'of', dataset.shape[0])

        baseline_tags = load_dicom_tagssafely(row['PRIOR_PATH'], prefix='baseline_')
        followup_tags = load_dicom_tagssafely(row['SUBSQ_PATH'], prefix='followup_')

        baseline_im = loader(row['PRIOR_PATH_NRRD'])
        followup_im = loader(row['SUBSQ_PATH_NRRD'])

        # +++
        #import SimpleITK as sitk
        #sitk.WriteImage(sitk.GetImageFromArray(baseline_im.squeeze()), 'baseline_im.nii.gz')
        #sitk.WriteImage(sitk.GetImageFromArray(followup_im.squeeze()), 'followup_im.nii.gz')
        # +++

        baseline_im = np2torch(baseline_im).to(device)
        followup_im = np2torch(followup_im).to(device)

        # compute embedding
        (_, wD), (_, tD) = registration_network(baseline_im, followup_im)
        tD_, wD_, Z = student_network(baseline_im, followup_im, return_embedding=True)
        z_fixed, z_moving, z_diff = Z

        z_diff = z_diff.detach().cpu().numpy().squeeze()
        features = array_to_dict(z_diff, 'z')

        z_moving = z_moving.detach().cpu().numpy().squeeze()
        features.update(array_to_dict(z_moving, 'z_moving'))

        z_fixed = z_fixed.detach().cpu().numpy().squeeze()
        features.update(array_to_dict(z_fixed, 'z_fixed'))

        # compute error metrics
        registration_loss           = correlation(baseline_im, wD)
        deformation_energy          = energy(tD)
        estimate_divergence         = correlation(tD, tD_)
        estimate_registration_loss  = correlation(baseline_im, wD_)

        # +++
        #import SimpleITK as sitk
        #temp = wD.detach().cpu().numpy().squeeze()
        #sitk.WriteImage(sitk.GetImageFromArray(temp.squeeze()), 'WD.nii.gz')
        # +++

        # store results
        entry = {
            'pair_index':                   i,
            'registration_loss':            registration_loss.item(),
            'deformation_energy':           deformation_energy.item(),
            'estimate_divergence':          estimate_divergence.item(),
            'estimate_registration_loss':   estimate_registration_loss.item()
        }
        entry.update(features)
        entry.update(baseline_tags)
        entry.update(followup_tags)

        result.append(entry)

    pd.DataFrame(result).to_csv('results.csv')

        # +++
        #break
        # +++

if __name__ == "__main__":

    cuda_seeds()
    reg_net, std_net, device = load_trained_models()

    with torch.no_grad():
        test(reg_net, std_net, PARAMS.inference, device)



