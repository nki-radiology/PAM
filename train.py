


import os
import shutil
import pandas as pd
import numpy as np
import wandb
import torch
import torch.nn as nn
import SimpleITK as sitk

from torch.utils.data               import DataLoader
from utils.utils_torch              import weights_init
from sklearn.model_selection        import train_test_split
from pathlib                        import Path
from datetime                       import datetime

from PAMDataset                     import PAMDataset
from PAMDataset                     import get_num_classes

from networks.PAMNetwork            import RegistrationNetwork
from networks.DiscriminatorNetwork  import DiscriminatorNetwork
from networks.PAMNetwork            import SegmentationNetwork    
from networks.PAMNetwork            import StudentNetwork

from metrics.PAMLoss                import correlation_coefficient_loss
from metrics.PAMLoss                import variatinal_energy_loss
from metrics.PAMLoss                import dice_loss

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


def model_init():
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Registration teacher
    reg_net = RegistrationNetwork(  PARAMS.img_dim, PARAMS.filters)
    reg_net.to(device)
    reg_net.apply(weights_init)

    dis_net = DiscriminatorNetwork( PARAMS.img_dim, PARAMS.filters_discriminator)
    dis_net.to(device)
    dis_net.apply(weights_init)

    # Segmentation teacher
    seg_net = SegmentationNetwork(  PARAMS.img_dim, PARAMS.filters, get_num_classes(PARAMS.body_part))
    seg_net.to(device)
    seg_net.apply(weights_init)

    # Student
    stu_net = StudentNetwork(  PARAMS.img_dim, PARAMS.filters, get_num_classes(PARAMS.body_part), PARAMS.latent_dim)
    stu_net.to(device)
    stu_net.apply(weights_init)

    networks = [reg_net, dis_net, seg_net, stu_net]

    return networks, device


class Trainer():
    def __init__(self, model, device, backup_path):
        self.model          = model
        self.device         = device
        self.backup_path    = backup_path

        self.itr            = 0

        self.load_backup_if_present()

    def save(self):
        torch.save(self.model.state_dict(), self.backup_path)

    def load_backup_if_present(self):
        if os.path.exists(self.backup_path):
            shutil.copyfile(self.backup_path, self.backup_path + '.backup')
            self.model.load_state_dict(torch.load(self.backup_path))
        print('backup loaded:', self.backup_path, '.')

    def inc_iterator(self):
        self.itr += 1


class RegistrationNetworkTrainer(Trainer):
    def __init__(self, model, discriminator, device, backup_path):
        super().__init__(model, device, backup_path)
        self.discriminator  = discriminator

        self.correlation_fn = correlation_coefficient_loss
        self.energy_fn      = variatinal_energy_loss
        self.mse_fn         = nn.MSELoss()

        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))
        self.model.train()


    def train(self, fixed, moving):
        self.inc_iterator()
        self.optimizer.zero_grad()

        # forward pass
        (wA, wD), (tA, tD) = self.model(fixed, moving)

        # adversarial loss
        # we use the affine as real and the elastic as fake
        _, features_wA      = self.discriminator(wA) 
        _, features_wD      = self.discriminator(wD) 
        adv_loss            = self.mse_fn(features_wA, features_wD)

        # registration loss
        reg_affine_loss   = self.correlation_fn(fixed, wA)
        reg_deform_loss   = self.correlation_fn(fixed, wD)

        # energy-like penalty loss
        energy_loss = self.energy_fn(tA) + self.energy_fn(tD)

        def fact(start=0, stop=1000.):
            factor = (self.itr - start)/stop
            factor = np.maximum(factor, 0.0)
            factor = np.minimum(factor, 1.0)
            return factor

        # total loss            
        loss = \
            1.0     * reg_affine_loss + \
            1.0     * reg_deform_loss + \
            0.1     * fact(self.itr, start=1000, stop=10000) * adv_loss + \
            0.01    * fact(self.itr, start=1000, stop=10000) * energy_loss 
        
        loss.backward()
        self.optimizer.step()

        return reg_affine_loss, reg_deform_loss, adv_loss, energy_loss


class DiscriminatorNetworkTrainer(Trainer):
    def __init__(self, model, device, backup_path):
        super().__init__(model, device, backup_path)
        self.model          = model
        self.device         = device
        self.real_label     = 1.

        self.bce_fn         = nn.BCELoss()
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))
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
            self.optimizer.apply(weights_init)
            print("Reloading discriminator weights")

        return loss
    

class SegmentationNetworkTrainer(Trainer):
    def __init__(self, model, device, backup_path):
        super().__init__(model, device, backup_path)

        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))
        self.dice_loss_fn   = dice_loss
        self.xent_loss_fn   = nn.CrossEntropyLoss()
        self.model.train()

    def train(self, image, target):
        self.inc_iterator()
        self.optimizer.zero_grad()

        # forward pass
        predicted = self.model(image)

        # segmentation loss
        loss    = self.dice_loss_fn(target, predicted)
        loss   += self.xent_loss_fn(target, predicted)

        loss.backward()
        self.optimizer.step()

        return loss


class StudentNetworkTrainer(Trainer):
    def __init__(self, model, device, backup_path):
        super().__init__(model, device, backup_path)

        self.correlation_fn = correlation_coefficient_loss
        self.mse_fn         = nn.MSELoss()
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))

    def train(self, 
              fixed, 
              moving, 
              registration_outputs, 
              segmentation_outputs,
              segmentation_targets
        ):
        self.inc_iterator()
        self.optimizer.zero_grad()

        # forward pass
        (w, t), (s_fixed, s_moving) = self.model(fixed, moving)

        # registration loss
        tA, tD                  = registration_outputs
        reg_consistency_loss    = self.mse_fn(tA + tD, t)
        reg_loss                = self.correlation_fn(fixed, w)

        # segmentation loss
        prob_fixed, prob_moving = segmentation_outputs
        seg_consistency_loss    = self.mse_fn(prob_fixed, s_fixed)
        seg_consistency_loss   += self.mse_fn(prob_moving, s_moving)

        mask_fixed, mask_moving = segmentation_targets
        dice_loss               = dice_loss(mask_fixed, s_fixed)
        dice_loss              += dice_loss(mask_moving, s_moving)

        loss = \
            1.0     * reg_loss + \
            0.001   * reg_consistency_loss + \
            0.5     * dice_loss + \
            0.01    * seg_consistency_loss
        
        loss.backward()
        self.optimizer.step()

        return (reg_loss, dice_loss), (reg_consistency_loss, seg_consistency_loss)


def training(
        registration_network,
        segmentation_network,
        student_network, 
        discriminator, 
        train_dataloader, 
        device
    ):
    
    epoch        = 0
    n_epochs     = 10001

    # wandb Initialization
    wandb.init(project=PARAMS.wandb, entity='s-trebeschi')
    wandb.watch(registration_network, log=None)

    registration_trainer    = RegistrationNetworkTrainer(
        registration_network, discriminator, device, os.path.join(PARAMS.project_folder, 'RegNet.pth'))
    discriminator_trainer   = DiscriminatorNetworkTrainer(
        discriminator, device, os.path.join(PARAMS.project_folder, 'DiscNet.pth'))
    segmentation_trainer    = SegmentationNetworkTrainer(
        segmentation_network, device, os.path.join(PARAMS.project_folder, 'SegNet.pth'))
    student_trainer         = StudentNetworkTrainer(
        student_network, device, os.path.join(PARAMS.project_folder, 'StuNet.pth'))

    for epoch in range(epoch, n_epochs):

        for _, (x_1, x_2) in enumerate(train_dataloader):
            # data loading
            fixed, fixed_mask   = x_1
            moving, moving_mask = x_2

            fixed  = fixed.to(device)
            moving = moving.to(device)

            fixed_mask  = fixed_mask.to(device)
            moving_mask = moving_mask.to(device)

            # registration training
            L = registration_trainer.train(fixed, moving)
            registration_affine_loss, registration_elastic_loss, adversarial_loss, energy_loss = L

            # discriminator training
            (wA, wD), (tA, tD) = registration_network(fixed, moving)
            L = discriminator_trainer.train(wA.detach(), wD.detach())
            discriminator_loss = L

            # segmentation training
            L = segmentation_trainer.train(fixed, fixed_mask)
            segmentation_loss_fixed = L

            L = segmentation_trainer.train(moving, moving_mask)
            segmentation_loss_moving = L

            # student training
            fixed_mask_pred     = segmentation_network(fixed)
            moving_mask_pred    = segmentation_network(moving)

            registration_outputs = (tA.detach(), tD.detach())
            segmentation_outputs = (fixed_mask_pred.detach(), moving_mask_pred.detach())
            segmentation_targets = (fixed_mask, moving_mask)

            L = student_trainer.train(fixed, moving, registration_outputs, segmentation_outputs, segmentation_targets)
            student_loss, student_consistency_loss = L

            student_registration_loss = student_loss[0]
            student_segmentation_loss = student_loss[1]

            student_registration_constistency_loss = student_consistency_loss[0]
            student_segmentation_constistency_loss = student_consistency_loss[1]

            # wandb logging
            wandb.log({ 
                'Train: Registration Similarity Affine loss':   registration_affine_loss.item(),
                'Train: Registration Similarity Elastic loss':  registration_elastic_loss.item(),
                'Train: Registration Energy loss':              energy_loss.item(),
                'Train: Registration Adversarial Loss':         adversarial_loss.item(),
                'Train: Registration Discriminator Loss':       discriminator_loss.item(),

                'Train: Segmentation Loss Fixed':               segmentation_loss_fixed.item(),
                'Train: Segmentation Loss Moving':              segmentation_loss_moving.item(),

                'Train: Student Segmentation Consistency Loss': student_segmentation_constistency_loss.item(),
                'Train: Student Segmentation Loss':             student_segmentation_loss.item(),
                'Train: Student Registration Consistency Loss': student_registration_constistency_loss.item(),
                'Train: Student Registration Loss':             student_registration_loss.item()
            })
            
        # Save checkpoints
        if (epoch % 5 == 0) and (epoch > 0):

            registration_trainer.save()
            discriminator_trainer.save()
            segmentation_trainer.save()
            student_trainer.save()

            print('Model saved!')


if __name__ == "__main__":
    
    cuda_seeds()

    breakpoint()

    networks, device    = model_init()
    train_dataloader, _ = data_init()

    registration_network, discriminator_network, segmentation_network, student_network = networks
    
    training(registration_network, segmentation_network, student_network, discriminator_network, device)