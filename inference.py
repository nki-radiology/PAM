import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from pydicom                import dcmread

from networks.PAMNetwork    import PAMNetwork
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
    discriminator_loss = nn.BCELoss()  
    l2_loss = nn.MSELoss()  
    nn_loss = Cross_Correlation_Loss().pearson_correlation
    penalty = Energy_Loss().energy_loss
    return discriminator_loss, l2_loss, nn_loss, penalty    


def load_model_weights():
    # Network definition
    pam_net     = PAMNetwork(PARAMS.img_dim, PARAMS.filters)
    device      = torch.device('cuda:0')
    pam_net.to(device)

    # Loading the model weights
    pam_chkpt = os.path.join(PARAMS.project_folder, 'PAMModel.pth')
    pam_net.load_state_dict(torch.load(pam_chkpt))

    return pam_net, device


def load_dicom_tagssafely(self, path, prefix = ''):
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


def test(pam_network, dataset, device):

    dataset = pd.read_csv(dataset, index_col=0)
    print('loaded dataset', str(dataset.shape[0]), 'rows')

    loader = ImageLoader(
        ReadVolume(),
        ZeroOneScaling(),
        TransformFromNumpyFunction(zero_at_edges),
        ToNumpyArray(
            add_batch_dim=True, 
            add_singleton_dim=True, 
            channel_second=True)
    )

    _, _, cc_loss, penalty = init_loss_functions()
    pam_network.eval()

    result = []

    for i, row in dataset.iterrows():

        print('processing row', i, 'of', dataset.shape[0])

        baseline_tags = load_dicom_tagssafely(row['PRIOR_PATH'], prefix='baseline_')
        followup_tags = load_dicom_tagssafely(row['SUBSQ_PATH'], prefix='followup_')

        baseline_im = loader(row['PRIOR_PATH_NRRD'])
        followup_im = loader(row['SUBSQ_PATH_NRRD'])

        baseline_im = baseline_im.to(device)
        followup_im = followup_im.to(device)

        # compute embedding
        z, residual = pam_network.get_features(baseline_im, followup_im)

        z = z.detach().cpu().numpy()
        residual = residual.detach().cpu().numpy()

        features = array_to_dict(z, 'features')
        residual = array_to_dict(residual, 'residual')

        # compute loss
        _, _, tD, wD, _ = pam_network(baseline_im, followup_im)
        loss = cc_loss(baseline_im, wD)
        energy = penalty(tD)

        # store results
        entry = {
            'i': i,
            'loss': loss.item(),
            'energy': energy.item()
        }
        entry.update(features)
        entry.update(residual)
        entry.update(baseline_tags)
        entry.update(followup_tags)

        result.append(entry)

        pd.DataFrame(result).to_csv('results.csv')


if __name__ == "__main__":

    cuda_seeds()
    pam_network, device = load_model_weights()

    with torch.no_grad():
        test(pam_network, PARAMS.inference, device)



