## Usage example
# python register.py 
#   --body-part thorax 
#   --dataset-followup /data/groups/beets-tan/s.trebeschi/MPM_FOLLOWUP/followup-dataset.csv 
#   --project-folder /projects/split-encoders/thorax-just-registration/ 
#   --output-folder /data/groups/beets-tan/s.trebeschi/MPM_FOLLOWUP/2.registrations/ 
#   --debug True 
#   --keep-network-size True

import os
import torch

from torch.utils.data               import DataLoader

from modules.Data                   import FollowUpDataset
from modules.Data                   import save_image
from modules.Networks               import RegistrationNetwork

from config                         import PARAMS

DEBUG                               = PARAMS.debug
PROJECT                             = PARAMS.project_folder
OUTPUT_FOLDER                       = PARAMS.output_folder
IMG_DIM                             = PARAMS.img_dim
FILTERS                             = PARAMS.filters
LATENT_DIM                          = PARAMS.latent_dim

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


def hardware_init():
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # where to store the model
    model_path = os.path.join(PROJECT,  'Network.pth')

    return model_path, device


def data_init():
    dataloader = DataLoader(
        dataset=FollowUpDataset(), 
        batch_size=1, 
        shuffle=False, 
        pin_memory=False
    )
    return dataloader


def load_registration_model(model_path, device):
    model = RegistrationNetwork(IMG_DIM, FILTERS)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def register(model, dataloader, device):
    for i, batch in enumerate(dataloader):
        print(f'Processing record {i}')
        baseline, followup = batch

        baseline    = baseline.to(device)
        followup    = followup.to(device)

        # forward pass
        with torch.no_grad():
            output = model(baseline, followup)
            (wA, wD), (_, tD)  = output

        # save output
        save_image(wA, os.path.join(OUTPUT_FOLDER, f'wA_{i}.nrrd'))
        #save_image(tA, os.path.join(PROJECT, f'tA_{i}.nrrd'))
        save_image(wD, os.path.join(OUTPUT_FOLDER, f'wD_{i}.nrrd'))
        save_image(tD, os.path.join(OUTPUT_FOLDER, f'tD_{i}.nrrd'))


if __name__ == '__main__':

    if DEBUG:
        breakpoint()

    cuda_seeds()

    model_path, device          = hardware_init()
    train_dataloader            = data_init()

    model                       = load_registration_model(model_path, device)

    register(model, train_dataloader, device)