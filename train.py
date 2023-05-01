


import os
import datetime
import pandas as pd
import numpy as np
import wandb
import torch
import torch.nn as nn

from torch.utils.data               import DataLoader
from utils.utils_torch              import weights_init
from sklearn.model_selection        import train_test_split
from pathlib                        import Path


from RegistrationDataset            import RegistrationDataSet
from networks.PAMNetwork            import PAMNetwork
from networks.DiscriminatorNetwork  import DiscriminatorNetwork
from metrics.LossPam                import Energy_Loss, Cross_Correlation_Loss
from metrics.HessianPenalty         import hessian_penalty

from config import PARAMS

RANDOM_SEED = 42


def read_train_data():
    path_input= PARAMS.train_folder
    path      = Path(path_input)
    filenames = list(path.glob('*.nrrd'))
    data_index= []

    for f in filenames:
        data_index.append(int(str(f).split('/')[7].split('_')[0])) 

    train_data = list(zip(data_index, filenames))
    train_data = pd.DataFrame(train_data, columns=['tcia_idx', 'dicom_path'])

    return train_data


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

    # Network Definitions to the device
    pam_net = PAMNetwork(PARAMS.img_dim, PARAMS.filters, PARAMS.latent_dim)
    dis_net = DiscriminatorNetwork(PARAMS.img_dim, PARAMS.filters_discriminator)
    pam_net.to(device)
    dis_net.to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        pam_net = nn.DataParallel(pam_net, list(range(ngpu)))
        dis_net = nn.DataParallel(dis_net, list(range(ngpu)))

    # Init weights for Generator and Discriminator
    pam_net.apply(weights_init)
    dis_net.apply(weights_init)

    return pam_net, dis_net, device


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


def get_optimizers(pam_net, dis_net):
    pam_optimizer = torch.optim.Adam(pam_net.parameters(), lr = 3e-4, betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(dis_net.parameters(), lr = 3e-4, betas=(0.5, 0.999))
    return pam_optimizer, dis_optimizer


def load_dataloader():
    filenames   = read_train_data()

    inputs_train, inputs_valid = train_test_split(
        filenames, random_state=RANDOM_SEED, train_size=0.8, shuffle=True
    )

    print("total: ", len(filenames), " train: ", len(inputs_train), " valid: ", len(inputs_valid))

    train_dataset = RegistrationDataSet(path_dataset = inputs_train,
                                        input_shape  = (300, 192, 192, 1),
                                        transform    = None)

    valid_dataset = RegistrationDataSet(path_dataset = inputs_valid,
                                        input_shape  = (300, 192, 192, 1),
                                        transform    = None)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=PARAMS.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=PARAMS.batch_size, shuffle=True)

    return train_dataloader, valid_dataloader


def training(
        pam_network, discriminator, 
        train_dataloader, 
        device
    ):
    
    epoch        = 0
    n_epochs     = 10001

    real_label   = 1.
    fake_label   = 0.

    (correlation, energy), (binary_entropy, mse_distance), (l2_norm, l1_norm) = init_loss_functions()
    pam_network_optimizer, discriminator_optimizer = get_optimizers(pam_network, discriminator_network)

    # wandb Initialization
    wandb.init(project=PARAMS.wandb, entity='s-trebeschi')
    wandb.watch(pam_network, log=None)

    pam_network.train()
    discriminator.train()

    for epoch in range(epoch, n_epochs):

        for i, (x_1, x_2) in enumerate(train_dataloader):

            # send to device (GPU or CPU)
            fixed  = x_1.to(device)
            moving = x_2.to(device)

            # *** Train Generator ***
            pam_network_optimizer.zero_grad()
            _, w_0, t_1, w_1, Z = pam_network(fixed, moving)
            z, residual = Z

            # adversarial loss
            # we use the affine as real and the elastic as fake
            _, features_w1      = discriminator(w_1) 
            _, features_w0      = discriminator(w_0) 
            generator_adv_loss  = mse_distance(features_w1, features_w0)

            # registration loss
            registration_affine_loss = correlation(fixed, w_0)
            registration_deform_loss = correlation(fixed, w_1)

            # circuit residual loss
            residual_loss       = l2_norm(residual)

            # energy-like penalty loss
            enegry_deformation  = energy(t_1)
            sparsity_loss       = l1_norm(z)

            # hessian loss
            hessian_loss        = hessian_penalty(pam_network.affine_decoder, z)
            hessian_loss       += hessian_penalty(pam_network.elastic_decoder, z)

            # total loss            
            loss = \
                1.0     * registration_affine_loss + \
                1.0     * registration_deform_loss + \
                0.01    * generator_adv_loss + \
                0.01    * enegry_deformation + \
                0.001   * residual_loss + \
                0.0001  * sparsity_loss + \
                0.0001  * hessian_loss
            
            loss.backward()
            pam_network_optimizer.step()

            # *** Train Discriminator ***
            discriminator_optimizer.zero_grad()

            real, _ = discriminator(w_0.detach()) 
            fake, _ = discriminator(w_1.detach())

            b_size   = real.shape
            label_r  = torch.full(b_size, real_label, dtype=torch.float, device=device)
            label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=device)

            loss_d_real = binary_entropy(real, label_r)
            loss_d_fake = binary_entropy(fake, label_f)
            loss_d_t    = (loss_d_real + loss_d_fake) * 0.5

            # -- accu_loss_discriminator += loss_d_t.item()
            loss_d_t.backward()
            discriminator_optimizer.step()

            # Reinit the affine network weights
            if loss_d_t.item() < 1e-5:  # >
                discriminator.apply(weights_init)
                print("Reloading discriminator weights")

            # Display in tensorboard
            # ========
            wandb.log({ 'Train: Similarity Affine loss': registration_affine_loss.item(),
                        'Train: Similarity Elastic loss': registration_deform_loss.item(),
                        'Train: Energy loss': enegry_deformation.item(),
                        'Train: Sparsity loss': sparsity_loss.item(),
                        'Train: Hessian loss': hessian_loss.item(),
                        'Train: Residual Loss': residual_loss.item(),
                        'Train: Adversarial Loss': generator_adv_loss.item(),
                        'Train: Total loss': loss.item(),
                        'Train: Discriminator Loss': loss_d_t.item()})
            
        # Save checkpoints
        if (epoch % 25 == 0) and (epoch > 0):
            name_pam = 'PAMModel.pth'
            name_dis = 'DisModel.pth'
            torch.save(pam_network.state_dict(), os.path.join(PARAMS.project_folder, name_pam))
            torch.save(discriminator_network.state_dict(), os.path.join(PARAMS.project_folder, name_dis))
            print('Model saved!')
            

def are_models_trained():
    name_pam = os.path.join(PARAMS.project_folder, 'PAMModel.pth')
    name_dis = os.path.join(PARAMS.project_folder, 'DisModel.pth')
    return os.path.exists(name_pam) and os.path.exists(name_dis)


def backup_existing_checkpoints():
    name_pam = os.path.join(PARAMS.project_folder, 'PAMModel.pth')
    os.copyfile(name_pam, os.path.join(PARAMS.project_folder, 'PAMModel.pth.bak-' + datetime.now().strftime("%Y/%m/%d/-%H:%M")))
    name_dis = os.path.join(PARAMS.project_folder, 'DisModel.pth')
    os.copyfile(name_dis, os.path.join(PARAMS.project_folder, 'DisModel.pth.bak-' + datetime.now().strftime("%Y/%m/%d/-%H:%M")))
    

def load_trained_models():
    # Network definition
    pam_net     = PAMNetwork(PARAMS.img_dim, PARAMS.filters)
    dis_net     = DiscriminatorNetwork(PARAMS.img_dim, PARAMS.filters)
    device      = torch.device('cuda:0')
    pam_net.to(device)

    # Loading the model weights
    pam_chkpt = os.path.join(PARAMS.project_folder, 'PAMModel.pth')
    dis_chkpt = os.path.join(PARAMS.project_folder, 'DisModel.pth')
    pam_net.load_state_dict(torch.load(pam_chkpt))
    dis_net.load_state_dict(torch.load(dis_chkpt))

    return pam_net, dis_net, device


if __name__ == "__main__":
    
    cuda_seeds()
    
    if are_models_trained():
        print("Models already trained. Backing up existing checkpoints.")
        backup_existing_checkpoints()
        pam_network, discriminator_network, device  = load_trained_models()
    else:
        pam_network, discriminator_network, device  = model_init()

    train_dataloader, _  = load_dataloader()
    training(pam_network, discriminator_network, train_dataloader, device)
