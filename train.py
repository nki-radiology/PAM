import os
import pandas as pd
import wandb
import torch

from torch.utils.data               import DataLoader
from sklearn.model_selection        import train_test_split
from pathlib                        import Path

from PAMDataset                     import PAMDataset
from Trainer                        import RegistrationNetworkTrainer, StudentNetworkTrainer

from config import PARAMS

RANDOM_SEED = 42


def read_train_data(load_segmentations = False):
    path_im   = PARAMS.train_folder

    def list_files(path, extension = '*.nrrd'):
        path      = Path(path)
        filenames = list(path.glob(extension))

        # iterate to get the TCIA index value
        # example: 0011_FILENAME.nrrd > 0011
        data_index= []
        for f in filenames:
            data_index.append(int(str(f).split('/')[-1].split('_')[0])) 

        train_data = list(zip(data_index, filenames))
        train_data = pd.DataFrame(train_data, columns=['tcia_idx', 'dicom_path'])
        return train_data
    
    train_data = list_files(path_im, '*.nrrd')

    if load_segmentations:
        path_seg  = PARAMS.train_folder_segmentations
        train_data = train_data.merge(list_files(path_seg, '*.nii.gz'), on='tcia_idx', suffixes=('','_seg'))
    
    return train_data


def data_init(load_segmentations = False):
    filenames   = read_train_data(load_segmentations)

    inputs_train, inputs_valid = train_test_split(
        filenames, random_state=RANDOM_SEED, train_size=0.8, shuffle=True
    )

    print("total: ", len(filenames), " train: ", len(inputs_train), " valid: ", len(inputs_valid))

    train_dataset = PAMDataset(dataset       = inputs_train,
                                input_shape  = (300, 192, 192, 1),
                                transform    = None,
                                body_part    = PARAMS.body_part)

    valid_dataset = PAMDataset(dataset       = inputs_valid,
                                input_shape  = (300, 192, 192, 1),
                                transform    = None,
                                body_part    = PARAMS.body_part)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=PARAMS.batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=PARAMS.batch_size, shuffle=True)

    return train_dataloader, valid_dataloader


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
    model_path = os.path.join(PARAMS.project_folder,  'Network.pth')

    return model_path, device


def save_image(image, path):
    from SimpleITK import GetImageFromArray
    from SimpleITK import WriteImage
    from numpy import transpose, squeeze

    image = image.cpu().detach().numpy()
    image = transpose(image, (0, 2, 3, 4, 1))
    image = squeeze(image)
    image = GetImageFromArray(image)
    WriteImage(image, path)


def training(
        trainer,
        train_dataloader, 
        device
    ):
    
    epoch        = 0
    n_epochs     = 10001

    # wandb Initialization
    if not PARAMS.debug:
        wandb.init(project=PARAMS.wandb, entity='s-trebeschi')

    for epoch in range(epoch, n_epochs):
        for _, (x_1, x_2) in enumerate(train_dataloader):
            # data loading
            fixed, fixed_mask   = x_1
            moving, moving_mask = x_2

            fixed               = fixed.to(device)
            moving              = moving.to(device)

            #fixed_mask          = fixed_mask.to(device)
            #moving_mask         = moving_mask.to(device)

            def add_noise(image):
                noise = torch.randn_like(image) / 10.
                return image + noise

            loss_dict           = trainer.train([add_noise(fixed), add_noise(moving)])
            # wandb logging
            if not PARAMS.debug:
                wandb.log(loss_dict)

            loss_dict           = trainer.train([add_noise(moving), add_noise(fixed)])
            # wandb logging
            if not PARAMS.debug:
                wandb.log(loss_dict)
            
        # Save checkpoints
        if not PARAMS.debug:
            trainer.save()
            print('Model saved!')

            (wA, wD), (tA, tD) = trainer.model(fixed, moving)

            print(tA)
            
            save_image(fixed,   os.path.join(PARAMS.project_folder, 'examples', 'fixed.nii.gz'))
            save_image(moving,  os.path.join(PARAMS.project_folder, 'examples', 'moving.nii.gz'))
            save_image(wA,      os.path.join(PARAMS.project_folder, 'examples', 'wA.nii.gz'))
            save_image(wD,      os.path.join(PARAMS.project_folder, 'examples', 'wD.nii.gz'))
            #save_image(tA,      os.path.join(PARAMS.project_folder, 'examples', 'tA.nii.gz'))
            save_image(tD,      os.path.join(PARAMS.project_folder, 'examples', 'tD.nii.gz'))


if __name__ == "__main__":

    if PARAMS.debug:
        breakpoint()
    else:
        print("Debug mode is off")
    
    cuda_seeds()

    model_path, device          = hardware_init()
    train_dataloader, _         = data_init(load_segmentations=True)

    if PARAMS.registration_only:
        trainer                 = RegistrationNetworkTrainer(device, model_path)
    else:
        trainer                 = StudentNetworkTrainer(device, model_path)

    training(
        trainer                 = trainer,
        train_dataloader        = train_dataloader, 
        device                  = device
    )