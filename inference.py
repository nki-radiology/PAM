import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from pydicom                import dcmread

from networks.PAMNetwork    import StudentNetwork
from PAMDataset             import get_num_classes

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


def load_trained_models():
    # Network definition
    std_net = StudentNetwork(  PARAMS.img_dim, PARAMS.filters, get_num_classes(PARAMS.body_part), PARAMS.latent_dim)
    device  = torch.device('cuda:0')

    def load_model(model, name):
        model.to(device)
        path = os.path.join(PARAMS.project_folder, name + '.pth')
        model.load_state_dict(torch.load(path))

    load_model(std_net, 'StdModel')

    return std_net, device


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


def test(student_network, dataset, device):

    dataset = pd.read_csv(dataset, index_col=0)
    print('loaded dataset', str(dataset.shape[0]), 'rows')

    loader = ImageLoader(
        ReadVolume(),
        ZeroOneScaling(),
        TransformFromNumpyFunction(zero_at_edges),
        ToNumpyArray(add_batch_dim=True, add_singleton_dim=True, channel_second=True)
    )

    student_network.eval()

    result = []

    for i, row in dataset.iterrows():

        print('processing row', i, 'of', dataset.shape[0])
        features = {}

        baseline_tags = load_dicom_tagssafely(row['PRIOR_PATH'], prefix='baseline_')
        features.update(baseline_tags)

        followup_tags = load_dicom_tagssafely(row['SUBSQ_PATH'], prefix='followup_')
        features.update(followup_tags)

        baseline_im = loader(row['PRIOR_PATH_NRRD'])
        followup_im = loader(row['SUBSQ_PATH_NRRD'])

        # +++
        if __debug__:
            import SimpleITK as sitk
            sitk.WriteImage(sitk.GetImageFromArray(baseline_im.squeeze()), 'baseline_im.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(followup_im.squeeze()), 'followup_im.nii.gz')
        # +++

        baseline_im = np2torch(baseline_im).to(device)
        followup_im = np2torch(followup_im).to(device)

        # compute embedding
        Z = student_network(baseline_im, followup_im, return_embedding=True)
        z_baseline, z_followup, z_diff = Z

        def append_embedding(embedding, dictionary, prefix):
            embedding = embedding.detach().cpu().numpy().squeeze()
            for i, value in enumerate(embedding):
                dictionary[f'{prefix}_{i}'] = value
            return dictionary
        
        features = {}
        features = append_embedding(z_diff,       features, 'feature')
        features = append_embedding(z_baseline,   features, 'feature_baseline')
        features = append_embedding(z_followup,   features, 'feature_followup')

        # +++
        if __debug__:
            import SimpleITK as sitk
            (w, t), (s_fixed, s_moving) = student_network(baseline_im, followup_im)
            sitk.WriteImage(sitk.GetImageFromArray(w.detach().cpu().numpy().squeeze()), 'w.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(t.detach().cpu().numpy().squeeze()), 't.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(s_fixed.detach().cpu().numpy().squeeze()), 's_fixed.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(s_moving.detach().cpu().numpy().squeeze()), 's_moving.nii.gz')
        # +++

        result.append(features)

    pd.DataFrame(result).to_csv('results.csv')

if __name__ == "__main__":

    if PARAMS.debug:
        breakpoint()
    else:
        print('no breakpoint set')

    cuda_seeds()
    reg_net, std_net, device = load_trained_models()

    with torch.no_grad():
        test(reg_net, std_net, PARAMS.inference, device)



