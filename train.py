


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

from RegistrationDataset            import RegistrationDataSet
from networks.PAMNetwork            import RegistrationNetwork, RegistrationStudentNetwork
from networks.DiscriminatorNetwork  import DiscriminatorNetwork
from metrics.LossPam                import Energy_Loss, Cross_Correlation_Loss

from config import PARAMS

RANDOM_SEED = 42


def read_train_data():
    path_input= PARAMS.train_folder
    path      = Path(path_input)
    filenames = list(path.glob('*.nrrd'))
    data_index= []

    for f in filenames:
        data_index.append(int(str(f).split('/')[-1].split('_')[0])) 

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
    reg_net = RegistrationNetwork(PARAMS.img_dim, PARAMS.filters)
    std_net = RegistrationStudentNetwork(PARAMS.img_dim, PARAMS.filters, PARAMS.latent_dim)
    dis_net = DiscriminatorNetwork(PARAMS.img_dim, PARAMS.filters_discriminator)

    reg_net.to(device)
    std_net.to(device)
    dis_net.to(device)

    # Init weights for Generator and Discriminator
    reg_net.apply(weights_init)
    std_net.apply(weights_init)
    dis_net.apply(weights_init)

    return reg_net, std_net, dis_net, device


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


def get_optimizers(reg_net, std_net, dis_net):
    reg_optimizer = torch.optim.Adam(reg_net.parameters(), lr = 3e-4, betas=(0.5, 0.999))
    std_optimizer = torch.optim.Adam(std_net.parameters(), lr = 3e-4, betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(dis_net.parameters(), lr = 3e-4, betas=(0.5, 0.999))

    return reg_optimizer, std_optimizer, dis_optimizer


def load_dataloader():
    filenames   = read_train_data()

    inputs_train, inputs_valid = train_test_split(
        filenames, random_state=RANDOM_SEED, train_size=0.8, shuffle=True
    )

    print("total: ", len(filenames), " train: ", len(inputs_train), " valid: ", len(inputs_valid))

    train_dataset = RegistrationDataSet(path_dataset = inputs_train,
                                        input_shape  = (300, 192, 192, 1),
                                        transform    = None,
                                        body_part    = PARAMS.body_part)

    valid_dataset = RegistrationDataSet(path_dataset = inputs_valid,
                                        input_shape  = (300, 192, 192, 1),
                                        transform    = None,
                                        body_part    = PARAMS.body_part)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=PARAMS.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=PARAMS.batch_size, shuffle=True)

    return train_dataloader, valid_dataloader


def training(
        registration_network,
        student_network, 
        discriminator, 
        train_dataloader, 
        device
    ):
    
    epoch        = 0
    n_epochs     = 10001

    real_label   = 1.
    fake_label   = 0.

    (correlation, energy), (binary_entropy, mse_distance), (_, _) = init_loss_functions()
    registration_opt, student_opt, discriminator_opt = get_optimizers(registration_network, student_network, discriminator_network)

    # wandb Initialization
    wandb.init(project=PARAMS.wandb, entity='s-trebeschi')
    wandb.watch(registration_network, log=None)

    registration_network.train()
    student_network.train()
    discriminator.train()

    for epoch in range(epoch, n_epochs):

        for i, (x_1, x_2) in enumerate(train_dataloader):

            # send to device (GPU or CPU)
            fixed  = x_1.to(device)
            moving = x_2.to(device)

            # *** Train Registration ***
            registration_opt.zero_grad()

            (wA, wD), (tA, tD) = registration_network(fixed, moving)

            # adversarial loss
            # we use the affine as real and the elastic as fake
            _, features_wA      = discriminator(wA) 
            _, features_wD      = discriminator(wD) 
            generator_adv_loss  = mse_distance(features_wA, features_wD)

            # registration loss
            registration_affine_loss = correlation(fixed, wA)
            registration_deform_loss = correlation(fixed, wD)

            # energy-like penalty loss
            enegry_deformation  = energy(tD) + energy(tA)

            # incremental factor for penalties and student loss
            itr = epoch * len(train_dataloader) + i

            def fact(itr, start=0, stop=1000.):
                factor = (itr - start)/stop
                factor = np.maximum(factor, 0.0)
                factor = np.minimum(factor, 1.0)
                return factor

            # total loss            
            loss = \
                1.0     * registration_affine_loss + \
                1.0     * registration_deform_loss + \
                0.1     * fact(itr, start=1000, stop=10000) * generator_adv_loss + \
                0.01    * fact(itr, start=1000, stop=10000) * enegry_deformation 
            
            loss.backward()
            registration_opt.step()

            # *** Train Student ***
            student_opt.zero_grad()
            student_t, student_w        = student_network(fixed, moving)
            student_consistency_loss    = mse_distance(student_t, tA.detach() + tD.detach())
            student_registration_loss   = correlation(fixed, student_w)

            student_loss = \
                1.0     * student_registration_loss + \
                0.001   * student_consistency_loss
            
            student_loss.backward()
            student_opt.step()

            # *** Train Discriminator ***
            discriminator_opt.zero_grad()

            real, _ = discriminator(wA.detach()) 
            fake, _ = discriminator(wD.detach())

            b_size   = real.shape
            label_r  = torch.full(b_size, real_label, dtype=torch.float, device=device)
            label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=device)

            loss_d_real = binary_entropy(real, label_r)
            loss_d_fake = binary_entropy(fake, label_f)
            loss_d_t    = (loss_d_real + loss_d_fake) * 0.5

            loss_d_t.backward()
            discriminator_opt.step()

            # Reinit the affine network weights
            if loss_d_t.item() < 1e-5:  # >
                discriminator.apply(weights_init)
                print("Reloading discriminator weights")

            # Display in tensorboard
            # ========
            wandb.log({ 'Train: Similarity Affine loss': registration_affine_loss.item(),
                        'Train: Similarity Elastic loss': registration_deform_loss.item(),
                        'Train: Energy loss': enegry_deformation.item(),
                        'Train: Adversarial Loss': generator_adv_loss.item(),
                        'Train: Student Consistency Loss': student_consistency_loss.item(),
                        'Train: Student Registration Loss': student_registration_loss.item(),
                        'Train: Total loss': loss.item(),
                        'Train: Discriminator Loss': loss_d_t.item()
            })
            
        # Save checkpoints
        if (epoch % 5 == 0) and (epoch > 0):
            def save_model(model, name):
                path = os.path.join(PARAMS.project_folder, name + '.pth')
                torch.save(model.state_dict(), path)

            save_model(registration_network, 'RegModel')
            save_model(student_network, 'StdModel')
            save_model(discriminator, 'DisModel')

            print('Model saved!')

        # Save example images
        if (epoch % 5 == 0) and (epoch > 0):

            def save_example_image(im, name):
                path = os.path.join(PARAMS.project_folder, name + '.nii.gz')
                sitk_im = sitk.GetImageFromArray(im.cpu().detach().numpy()[0,0,:,:,:])
                sitk.WriteImage(sitk_im, path)

            save_example_image(fixed, 'fixed')
            save_example_image(moving, 'moving')
            save_example_image(wD, 'test_deformable')
            save_example_image(wA, 'test_affine')          


def are_models_trained():
    name_reg = os.path.join(PARAMS.project_folder, 'RegModel.pth')
    name_std = os.path.join(PARAMS.project_folder, 'StdModel.pth')
    name_dis = os.path.join(PARAMS.project_folder, 'DisModel.pth')
    return os.path.exists(name_reg) and os.path.exists(name_std) and os.path.exists(name_dis)


def backup_existing_checkpoints():
    name_reg = os.path.join(PARAMS.project_folder, 'RegModel.pth')
    shutil.copyfile(name_reg, os.path.join(PARAMS.project_folder, 'RegModel.pth.bak'))
    name_std = os.path.join(PARAMS.project_folder, 'StdModel.pth')
    shutil.copyfile(name_std, os.path.join(PARAMS.project_folder, 'StdModel.pth.bak'))
    name_dis = os.path.join(PARAMS.project_folder, 'DisModel.pth')
    shutil.copyfile(name_dis, os.path.join(PARAMS.project_folder, 'DisModel.pth.bak'))
    

def load_trained_models():
    # Network definition
    reg_net     = RegistrationNetwork(PARAMS.img_dim, PARAMS.filters)
    std_net     = RegistrationStudentNetwork(PARAMS.img_dim, PARAMS.filters, PARAMS.latent_dim)
    dis_net     = DiscriminatorNetwork(PARAMS.img_dim, PARAMS.filters_discriminator)
    
    device      = torch.device('cuda:0')

    def load_model(model, name):
        model.to(device)
        path = os.path.join(PARAMS.project_folder, name + '.pth')
        model.load_state_dict(torch.load(path))

    load_model(reg_net, 'RegModel')
    load_model(std_net, 'StdModel')
    load_model(dis_net, 'DisModel')

    return reg_net, std_net, dis_net, device


if __name__ == "__main__":
    
    cuda_seeds()
    
    if are_models_trained():
        print("Models already trained. Backing up existing checkpoints.")
        backup_existing_checkpoints()
        registration_network, student_network, discriminator_network, device  = load_trained_models()
    else:
        registration_network, student_network, discriminator_network, device = model_init()

    train_dataloader, _  = load_dataloader()
    training(registration_network, student_network, discriminator_network, train_dataloader, device)