import os
import shutil
import pandas as pd
import numpy as np
import wandb
import torch
import torch.nn as nn

from utils.utils_torch              import weights_init
from torch.nn.functional            import affine_grid

from modules.Networks               import RegistrationNetwork
from modules.DiscriminatorNetwork   import DiscriminatorNetwork
from modules.Networks               import StudentNetwork

from modules.Losses                 import correlation_coefficient_loss
from modules.Losses                 import variatinal_energy_loss
from modules.Losses                 import orthogonal_loss

from config                         import PARAMS

IMG_DIM                             = PARAMS.img_dim
FILTERS_DISCRIMINATOR               = PARAMS.filters_discriminator
FILTERS                             = PARAMS.filters
LATENT_DIM                          = PARAMS.latent_dim

RANDOM_SEED = 42


class Trainer():
    def __init__(self, device, backup_path):
        self.device         = device
        self.backup_path    = backup_path
        self.itr            = 0

        self.init_model()        
        self.init_loss_funcions()
        self.init_optimizer()
        self.load_backup_if_present()


    def init_model(self):
        raise NotImplementedError
    

    def init_loss_funcions(self):
        raise NotImplementedError
    

    def init_optimizer(self):
        raise NotImplementedError
    

    def train(self, batch):
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


class DiscriminatorNetworkTrainer(Trainer):
    def __init__(self, device, backup_path):
        super().__init__(device, backup_path)
        self.real_label     = 1.


    def init_model(self):
        self.model = DiscriminatorNetwork(IMG_DIM, FILTERS_DISCRIMINATOR)
        self.model.to(self.device)
        self.model.apply(weights_init)
        self.model.train()


    def init_loss_funcions(self):
        self.bce_fn         = nn.BCELoss()
        self.mse_fn         = nn.MSELoss()


    def init_optimizer(self):
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))


    def train(self, batch):
        self.inc_iterator()
        self.optimizer.zero_grad()

        real, fake  = batch

        real, _     = self.model(real)
        fake, _     = self.model(fake)

        b_size      = real.shape
        label_r     = torch.full(b_size, self.real_label, dtype=torch.float, device=self.device)
        label_f     = torch.full(b_size, 1 - self.real_label, dtype=torch.float, device=self.device)

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
        _, features_real    = self.model(real) 
        _, features_fake    = self.model(fake) 
        loss                = self.mse_fn(features_real, features_fake)

        return loss


class RegistrationNetworkTrainer(Trainer):
    def __init__(self, device, backup_path):
        self.discriminator  = DiscriminatorNetworkTrainer(device, backup_path.replace('.pth', '_discriminator.pth'))
        super().__init__(device, backup_path)


    def init_model(self):
        self.model = RegistrationNetwork(IMG_DIM, FILTERS)
        self.model.to(self.device)
        self.model.apply(weights_init)
        self.model.train()

    
    def init_loss_funcions(self):
        self.correlation_fn = correlation_coefficient_loss

        self.mse_fn         = nn.MSELoss()
        self.adv_loss_fn    = self.discriminator.adversarial_loss

        self.orthogonal_fn  = orthogonal_loss
        self.variational_fn = variatinal_energy_loss


    def init_optimizer(self):
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))


    def train(self, batch):
        self.inc_iterator()
        self.optimizer.zero_grad()
        fixed, moving       = batch

        # forward pass
        (wA, wD), (tA, tD)  = self.model(fixed, moving)

        # registration loss
        # standard registration loss
        reg_affine_loss     = self.correlation_fn(fixed, wA)
        reg_deform_loss     = self.correlation_fn(fixed, wD)

        # energy-like penalty loss
        # make the transformation smooth
        penalty_affine      = self.orthogonal_fn(tA) 
        penalty_elastic     = self.variational_fn(tD) 

        # adversarial loss
        # make reigstreed image look like the fixed image
        real_image          = wA.detach()
        adv_loss            = self.adv_loss_fn(real_image, wD)

        loss = \
            3.0     * reg_affine_loss + \
            1.0     * reg_deform_loss + \
            0.5     * adv_loss + \
            0.00    * penalty_affine + \
            0.1     * penalty_elastic 
        
        loss.backward()
        self.optimizer.step()

        # train discriminator
        (wA, wD), (tA, tD)  = self.model(fixed, moving)
        L = self.discriminator.train([wA.detach(), wD.detach()])
        discriminator_loss = L

        # return losses
        loss_dict = {
            'total_loss':                   loss.item(),
            'reg_affine_loss':              reg_affine_loss.item(),
            'reg_deform_loss':              reg_deform_loss.item(),
            'adv_loss':                     adv_loss.item(),
            'penalty_affine':               penalty_affine.item(),
            'penalty_elastic':              penalty_elastic.item(),
            'discriminator_loss':           discriminator_loss.item(),
        }

        return loss_dict
    
    def save(self):
        super().save()
        self.discriminator.save()


class StudentNetworkTrainer(Trainer):
    def __init__(self, device, backup_path):
        self.registration     = RegistrationNetworkTrainer(device, backup_path.replace('.pth', '_registration.pth'))
        self.registration.model.eval()
        super().__init__(device, backup_path)


    def init_model(self):
        self.model = StudentNetwork(IMG_DIM, 1, FILTERS, LATENT_DIM)
        self.model.to(self.device)
        self.model.apply(weights_init)
        self.model.train()

    
    def init_loss_funcions(self):
        self.registration_loss  = correlation_coefficient_loss
        self.consistency_loss   = nn.MSELoss()


    def init_optimizer(self):
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = 3e-4, betas=(0.5, 0.999))


    def train(self, batch):
        self.inc_iterator()
        self.optimizer.zero_grad()
        fixed, moving           = batch

        # get registration target
        with torch.no_grad():
            _, (tA, tD)  = self.registration.model(fixed, moving)

        # merge deformation fields
        transform               = affine_grid(tA, moving.shape, align_corners=False)
        transform               = transform.permute(0, 4, 1, 2, 3)
        transform               = transform + tD

        # forward pass
        warped_pred, transform_pred  = self.model(fixed, moving)

        # registration loss
        registration_loss       = self.registration_loss(fixed, warped_pred)

        # consistency loss
        consistency_loss        = self.consistency_loss(transform_pred, transform)

        # total loss
        loss = \
            1.0     * registration_loss + \
            0.1     * consistency_loss
        
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            'total_loss':           loss.item(),
            'registration_loss':    registration_loss.item(),
            'consistency_loss':     consistency_loss.item(),
        }

        return loss_dict        

        

"""
def smooth_images(*images):
    results = []

    for i in images:
        i = nn.functional.avg_pool3d(i, kernel_size=3, stride=1, padding=1)
        results.append(i)
    return results


def weight_fn():
    w = np.sin(self.itr * np.pi / 1000.0)
    w = 0.1 if w < 0.1 else w
    return w


class SegmentationNetworkTrainer(Trainer):
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


class StudentNetworkTrainer(Trainer):
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


    def train(self, fixed, moving):
        self.inc_iterator()
        self.optimizer.zero_grad()

        fixed, fixed_mask   = fixed
        moving, moving_mask = moving

        # registration training
        _ = self.registration.train(fixed, moving)

        # segmentation training
        _ = self.segmentation.train(fixed, fixed_mask)
        _ = self.segmentation.train(moving, moving_mask)

        # student training
        (w, t), (s_fixed, s_moving) = self.model(fixed, moving)

        # registration loss
        (wA, _), (tA, tD)      = self.registration.model(fixed, moving)
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

        (w, t), (_, _)  = self.model(fixed, moving)
        L = self.discriminator.train(wA.detach(), w.detach())
        discriminator_loss = L

        loss_dict = {
            'reg_loss':                     reg_loss.item(),
            'dice_loss':                    dice_loss.item(),
            'reg_consistency_loss':         reg_consistency_loss.item(),
            'seg_consistency_loss':         seg_consistency_loss.item(),
            'discriminator_loss':           discriminator_loss.item(),
        }

        return loss_dict
    
    def save(self):
        super().save()
        self.registration.save()
        self.segmentation.save()
        self.discriminator.save()
"""