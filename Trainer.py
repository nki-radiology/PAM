import os
import shutil
import pandas as pd
import numpy as np
import wandb
import torch
import torch.nn as nn

from utils.utils_torch              import weights_init

from PAMDataset                     import get_num_classes

from networks.PAMNetwork            import RegistrationNetwork
from networks.PAMNetwork            import RegistrationNetworkV2
from networks.DiscriminatorNetwork  import DiscriminatorNetwork
from networks.PAMNetwork            import SegmentationNetwork    
from networks.PAMNetwork            import StudentNetwork

from metrics.PAMLoss                import correlation_coefficient_loss
from metrics.PAMLoss                import variatinal_energy_loss
from metrics.PAMLoss                import dice_loss
from metrics.PAMLoss                import xent_segmentation

from config import PARAMS

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


class DiscriminatorNetworkFactory(Trainer):
    def __init__(self, device, backup_path):
        super().__init__(device, backup_path)
        self.real_label     = 1.


    def init_model(self):
        self.model = DiscriminatorNetwork( PARAMS.img_dim, PARAMS.filters_discriminator)
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
        self.discriminator  = DiscriminatorNetworkFactory(device, backup_path.replace('.pth', '_discriminator.pth'))
        super().__init__(device, backup_path)


    def init_model(self):
        self.model = RegistrationNetworkV2(PARAMS.img_dim, PARAMS.filters)
        self.model.to(self.device)
        self.model.apply(weights_init)
        self.model.train()

    
    def init_loss_funcions(self):
        self.correlation_fn = correlation_coefficient_loss
        self.energy_fn      = variatinal_energy_loss
        self.mse_fn         = nn.MSELoss()
        self.adv_loss_fn    = self.discriminator.adversarial_loss


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
        energy_loss         = self.energy_fn(tA) + self.energy_fn(tD)

        # latent loss
        z_fixed             = self.model.encoder(fixed)
        z_wD                = self.model.encoder(wD)
        latent_loss         = self.mse_fn(z_fixed, z_wD)

        # adversarial loss
        # make reigstreed image look like the fixed image
        adv_loss            = self.adv_loss_fn(wA, wD)

        # total loss
        def fact(start=0, stop=1000.):
            factor = (self.itr - start)/stop
            factor = np.maximum(factor, 0.0)
            factor = np.minimum(factor, 1.0)
            return factor

        loss = \
            1.0     * reg_affine_loss + \
            1.0     * reg_deform_loss + \
            1.0     * fact(start=100, stop=1000) * adv_loss + \
            0.1     * fact(start=100, stop=1000) * energy_loss + \
            0.1     * fact(start=100, stop=1000) * latent_loss
        
        loss.backward()
        self.optimizer.step()

        # train discriminator
        (wA, wD), (tA, tD)  = self.model(fixed, moving)
        L = self.discriminator.train([wA.detach(), wD.detach()])
        discriminator_loss = L

        # return losses
        loss_dict = {
            'reg_affine_loss':              reg_affine_loss.item(),
            'reg_deform_loss':              reg_deform_loss.item(),
            'adv_loss':                     adv_loss.item(),
            'energy_loss':                  energy_loss.item(),
            'latent_loss':                  latent_loss.item(),
            'discriminator_loss':           discriminator_loss.item(),
        }

        return loss_dict
    
    def save(self):
        super().save()
        self.discriminator.save()


class StudentNetworkTrainer(Trainer):
    def __init__(self, device, backup_path):
        super().__init__(device, backup_path)
        raise NotImplementedError

"""
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