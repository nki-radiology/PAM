


import os
import shutil
import pandas as pd
import numpy as np
import wandb
import torch
import torch.nn as nn

from torch.utils.data               import DataLoader
from utils.utils_torch              import weights_init
from sklearn.model_selection        import train_test_split
from pathlib                        import Path

from PAMDataset                     import PAMDataset
from PAMDataset                     import get_num_classes

from networks.PAMNetwork            import RegistrationNetwork
from networks.DiscriminatorNetwork  import DiscriminatorNetwork
from networks.PAMNetwork            import SegmentationNetwork    
from networks.PAMNetwork            import StudentNetwork

from metrics.PAMLoss                import correlation_coefficient_loss
from metrics.PAMLoss                import variatinal_energy_loss
from metrics.PAMLoss                import dice_loss
from metrics.PAMLoss                import xent_segmentation

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

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=PARAMS.batch_size, shuffle=True)
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


class NetworkFactory():
    def __init__(self, device, backup_path):
        self.device         = device
        self.backup_path    = backup_path
        self.itr            = 0

        self.init_model()
        self.load_backup_if_present()


    def init_model(self):
        raise NotImplementedError


    def save(self):
        torch.save(self.model.state_dict(), self.backup_path)


    def load_backup_if_present(self):
        if os.path.exists(self.backup_path):
            shutil.copyfile(self.backup_path, self.backup_path + '.backup')
            self.model.load_state_dict(torch.load(self.backup_path))
            print('backup loaded:', self.backup_path, '.')
        else:
            print('no backup found:', self.backup_path, '.')


    def inc_iterator(self):
        self.itr += 1


class DiscriminatorNetworkFactory(NetworkFactory):
    def __init__(self, device, backup_path):
        super().__init__(device, backup_path)
        self.device         = device
        self.real_label     = 1.

        self.bce_fn         = nn.BCELoss()
        self.mse_fn         = nn.MSELoss()

        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))


    def init_model(self):
        self.model = DiscriminatorNetwork( PARAMS.img_dim, PARAMS.filters_discriminator)
        self.model.to(self.device)
        self.model.apply(weights_init)
        self.model.train()


    def train(self, real, fake):
        self.inc_iterator()
        self.optimizer.zero_grad()

        real, _ = self.model(real)
        fake, _ = self.model(fake)

        b_size = real.shape
        label_r = torch.full(b_size, self.real_label, dtype=torch.float, device=device)
        label_f = torch.full(b_size, 1 - self.real_label, dtype=torch.float, device=device)

        loss_d_real = self.bce_fn(real, label_r)
        loss_d_fake = self.bce_fn(fake, label_f)
        loss        = (loss_d_real + loss_d_fake) * 0.5

        loss.backward()
        self.optimizer.step()

        # Reinit the affine network weights
        if loss.item() < 1e-5:  # 
            self.model.apply(weights_init)
            print("Reloading discriminator weights")

        return loss
    

    def adversarial_loss(self, real, fake):
        _, features_real    = self.discriminator(real) 
        _, features_fake    = self.discriminator(fake) 
        loss                = self.mse_fn(features_real, features_fake)

        return loss


class RegistrationNetworkTrainer(NetworkFactory):
    def __init__(self, device, backup_path):
        super().__init__(device, backup_path)

        self.correlation_fn = correlation_coefficient_loss
        self.energy_fn      = variatinal_energy_loss
        self.mse_fn         = nn.MSELoss()

        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))

        self.discriminator  = DiscriminatorNetworkFactory(device, backup_path.replace('.pth', '_discriminator.pth'))
        self.adv_loss_fn    = self.discriminator.adversarial_loss


    def init_model(self):
        self.model = RegistrationNetwork( PARAMS.img_dim, PARAMS.filters)
        self.model.to(self.device)
        self.model.apply(weights_init)
        self.model.train()


    def train(self, fixed, moving):
        self.inc_iterator()
        self.optimizer.zero_grad()

        # forward pass
        (wA, wD), (tA, tD)  = self.model(fixed, moving)

        # adversarial loss
        # we use the affine as real and the elastic as fake
        adv_loss            = self.adv_loss_fn(wA, wD)

        # registration loss
        reg_affine_loss     = self.correlation_fn(fixed, wA)
        reg_deform_loss     = self.correlation_fn(fixed, wD)

        # energy-like penalty loss
        energy_loss         = self.energy_fn(tA) + self.energy_fn(tD)

        def fact(start=0, stop=1000.):
            factor = (self.itr - start)/stop
            factor = np.maximum(factor, 0.0)
            factor = np.minimum(factor, 1.0)
            return factor

        # total loss            
        loss = \
            1.0     * reg_affine_loss + \
            1.0     * reg_deform_loss + \
            1.0     * fact(start=1000, stop=10000) * adv_loss + \
            0.1     * fact(start=1000, stop=10000) * energy_loss 
        
        loss.backward()
        self.optimizer.step()

        (wA, wD), (tA, tD)  = self.model(fixed, moving)
        L = self.discriminator.train(wA.detach(), wD.detach())
        discriminator_loss = L

        return reg_affine_loss, reg_deform_loss, adv_loss, energy_loss, discriminator_loss
    
    def save(self):
        super().save()
        self.discriminator.save()


class SegmentationNetworkTrainer(NetworkFactory):
    def __init__(self, device, backup_path):
        super().__init__(device, backup_path)

        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))
        self.dice_loss_fn   = dice_loss
        self.xent_loss_fn   = xent_segmentation


    def init_model(self):
        self.model = SegmentationNetwork( PARAMS.img_dim, PARAMS.filters, get_num_classes(PARAMS.body_part))
        self.model.to(self.device)
        self.model.apply(weights_init)
        self.model.train()


    def train(self, image, target):
        self.inc_iterator()
        self.optimizer.zero_grad()

        # forward pass
        predicted = self.model(image)

        # segmentation loss
        loss    = self.dice_loss_fn(target, predicted)
        #loss   += self.xent_loss_fn(target, predicted)

        loss.backward()
        self.optimizer.step()

        return loss


class StudentNetworkTrainer(NetworkFactory):
    def __init__(self, device, backup_path):
        super().__init__(device, backup_path)

        self.correlation_fn = correlation_coefficient_loss
        self.mse_fn         = nn.MSELoss()
        self.dice_loss_fn   = dice_loss
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))

        self.discriminator  = DiscriminatorNetworkFactory(device, backup_path.replace('.pth', '_discriminator.pth'))
        self.adv_loss_fn    = self.discriminator.adversarial_loss

        self.registration   = RegistrationNetworkTrainer(device, backup_path.replace('.pth', '_registration.pth'))
        self.segmentation   = SegmentationNetworkTrainer(device, backup_path.replace('.pth', '_segmentation.pth'))


    def init_model(self):
        self.model = StudentNetwork( PARAMS.img_dim, PARAMS.filters, get_num_classes(PARAMS.body_part), PARAMS.latent_dim)
        self.model.to(self.device)
        self.model.apply(weights_init)
        self.model.train()


    def train(self, 
              fixed, 
              moving, 
              fixed_mask,
              moving_mask,
        ):
        self.inc_iterator()
        self.optimizer.zero_grad()

        # registration training
        L = self.registration.train(fixed, moving)

        # segmentation training
        L = self.segmentation.train(fixed, fixed_mask)
        L = self.segmentation.train(moving, moving_mask)

        # student training
        (w, t), (s_fixed, s_moving) = self.model(fixed, moving)

        # registration loss
        (wA, wD), (tA, tD)      = self.registration.model(fixed, moving)
        reg_consistency_loss    = self.mse_fn(tA.detach() + tD.detach(), t)
        reg_loss                = self.correlation_fn(fixed, w)
        adv_loss                = self.adv_loss_fn(wA, w)

        # segmentation loss
        prob_fixed              = self.segmentation.model(fixed)
        prob_moving             = self.segmentation.model(moving)

        seg_consistency_loss    = self.mse_fn(prob_fixed, s_fixed)
        seg_consistency_loss   += self.mse_fn(prob_moving, s_moving)

        dice_loss               = self.dice_loss_fn(fixed_mask, s_fixed)
        dice_loss              += self.dice_loss_fn(moving_mask, s_moving)

        def fact(start=0, stop=1000.):
            factor = (self.itr - start)/stop
            factor = np.maximum(factor, 0.0)
            factor = np.minimum(factor, 1.0)
            return factor

        loss = \
            1.0     * reg_loss + \
            0.001   * reg_consistency_loss + \
            1.0     * fact(start=1000, stop=10000) * adv_loss + \
            0.5     * dice_loss + \
            50      * seg_consistency_loss
        
        loss.backward()
        self.optimizer.step()

        (wA, wD), (tA, tD)  = self.model(fixed, moving)
        L = self.discriminator.train(wA.detach(), wD.detach())
        discriminator_loss = L

        return (reg_loss, dice_loss), (reg_consistency_loss, seg_consistency_loss), discriminator_loss
    
    def save(self):
        super().save()
        self.registration.save()
        self.segmentation.save()
        self.discriminator.save()


def hardware_init():
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # where to store the model
    model_path = os.path.join(PARAMS.project_folder,  'StudentNetwork.pth')

    return model_path, device


def training(
        student_network,
        train_dataloader, 
        device
    ):
    breakpoint()
    epoch        = 0
    n_epochs     = 10001

    # wandb Initialization
    #wandb.init(project=PARAMS.wandb, entity='s-trebeschi')


    for epoch in range(epoch, n_epochs):
        for _, (x_1, x_2) in enumerate(train_dataloader):
            # data loading
            fixed, fixed_mask   = x_1
            moving, moving_mask = x_2

            fixed  = fixed.to(device)
            moving = moving.to(device)

            fixed_mask  = fixed_mask.to(device)
            moving_mask = moving_mask.to(device)

            L = student_network.train(fixed, moving, fixed_mask, moving_mask)

            student_registration_loss                   = L[0]
            student_segmentation_loss                   = L[1]
            student_registration_constistency_loss      = L[2]
            student_segmentation_constistency_loss      = L[3]
            student_discriminator_loss                  = L[4]

            # wandb logging
            #wandb.log({ 
            #    'Train: Student Segmentation Consistency Loss': student_segmentation_constistency_loss.item(),
            #    'Train: Student Segmentation Loss':             student_segmentation_loss.item(),
            #    'Train: Student Registration Consistency Loss': student_registration_constistency_loss.item(),
            #    'Train: Student Registration Loss':             student_registration_loss.item(),
            #    'Train: Student Discriminator Loss':            student_discriminator_loss.item(),
            #})
            
        # Save checkpoints
        if (epoch % 5 == 0) and (epoch > 0):
            student_network.save()

            print('Model saved!')


if __name__ == "__main__":
    
    cuda_seeds()

    model_path, device          = hardware_init()
    train_dataloader, _         = data_init(load_segmentations=True)

    student_network             = StudentNetworkTrainer(device, model_path)

    training(
        student_network         = student_network,
        train_dataloader        = train_dataloader, 
        device                  = device
    )