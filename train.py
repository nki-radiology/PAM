# old 
import os
import wandb
import torch

from torch.utils.data           import DataLoader

from modules.Data               import RegistrationDataset
from modules.Data               import save_image

from modules.Trainer            import RegistrationNetworkTrainer
from modules.Trainer            import StudentNetworkTrainer

from config import PARAMS

BATCH_SIZE  = PARAMS.batch_size
DEBUG       = PARAMS.debug
MODULE      = PARAMS.module
WANDB       = PARAMS.wandb
PROJECT     = PARAMS.project_folder

RANDOM_SEED = 42

def data_init():
    dataloader = DataLoader(
        dataset=RegistrationDataset(), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True
    )
    return dataloader


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


def training(
        trainer,
        train_dataloader, 
        device
    ):
    
    epoch        = 0
    n_epochs     = 10001

    # wandb Initialization
    if not DEBUG:
        wandb.init(project=WANDB, entity='s-trebeschi')

    for epoch in range(epoch, n_epochs):
        for _, (x_1, x_2) in enumerate(train_dataloader):
            fixed, moving       = x_1, x_2
            fixed               = fixed.to(device)
            moving              = moving.to(device)

            def add_noise(image):
                noise = torch.randn_like(image) / 100.
                return image + noise

            loss_dict           = trainer.train([add_noise(fixed), add_noise(moving)])
            # wandb logging
            if not DEBUG:
                wandb.log(loss_dict)

            loss_dict           = trainer.train([add_noise(moving), add_noise(fixed)])
            # wandb logging
            if not DEBUG:
                wandb.log(loss_dict)
            
        # Save checkpoints
        if not DEBUG:
            trainer.save()
            print('Model saved!')

            warp, transform = trainer.model(fixed, moving)
            
            save_image(fixed,       os.path.join(PROJECT, 'examples', 'fixed.nii.gz'))
            save_image(moving,      os.path.join(PROJECT, 'examples', 'moving.nii.gz'))
            save_image(transform,   os.path.join(PROJECT, 'examples', 'transform.nii.gz'))
            save_image(warp,        os.path.join(PROJECT, 'examples', 'warp.nii.gz'))


if __name__ == "__main__":

    if DEBUG:
        breakpoint()
    else:
        print("Debug mode is off")
    
    cuda_seeds()

    model_path, device          = hardware_init()
    train_dataloader            = data_init()

    if MODULE == 'registration':
        trainer     = RegistrationNetworkTrainer(device, model_path)
    elif MODULE == 'student':
        trainer     = StudentNetworkTrainer(device, model_path)
    else:
        raise ValueError('Unknown module')

    training(
        trainer                 = trainer,
        train_dataloader        = train_dataloader, 
        device                  = device
    )