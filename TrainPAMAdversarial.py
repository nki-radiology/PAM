# Importing general lybraries
import os
import wandb
import pandas                     as     pd
import torch.nn                   as     nn
from   pathlib                    import Path
from   torch.utils.data           import DataLoader
from   sklearn.model_selection    import train_test_split

import torchvision.transforms     as     T
from   PIL                        import Image
# Importing model classes
from networks.PAMNetwork           import PAMNetwork
from networks.DiscriminatorNetwork import DiscriminatorNetwork

# Importing the dataset class
from RegistrationDataset          import RegistrationDataSet

# Importing helper functions
from utils.utils                  import create_directory
from utils.utils_torch            import weights_init
from utils.utils_torch            import draw_curve
from metrics.LossPam              import *
from config                       import args_pam_adv_fts_sit


def read_train_data():
    path_input= args_pam_adv_fts_sit.train_folder
    path      = Path(path_input)
    filenames = list(path.glob('*.nrrd'))
    data_index= []

    for f in filenames:
        data_index.append(int(str(f).split('/')[7].split('_')[0])) # Number 8 can vary according to the path of the images

    train_data = list(zip(data_index, filenames))
    train_data = pd.DataFrame(train_data, columns=['tcia_idx', 'dicom_path'])

    return train_data


def make_directory_to_save_chkp():
    dir_name = args_pam_adv_fts_sit.checkpoints_folder
    create_directory(dir_name)


def cuda_seeds():
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

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
    pam_net = PAMNetwork()
    dis_net = DiscriminatorNetwork()
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
    
    disc_loss   = nn.BCELoss()  # discriminator Loss
    l2_loss     = nn.MSELoss()  # Feature matching loss: fts_loss
    nn_loss     = Cross_Correlation_Loss() 
    energy_loss = Energy_Loss()


    return disc_loss, l2_loss, nn_loss, energy_loss


def get_optimizers(pam_net, dis_net):
    pam_optimizer = torch.optim.Adam(pam_net.parameters(), lr = 3e-4, betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(dis_net.parameters(), lr = 3e-4, betas=(0.5, 0.999))

    return pam_optimizer, dis_optimizer


def load_dataloader():
    # Dataset Path
    filenames   = read_train_data()

    # Random seed
    random_seed = 42

    # Split dataset into training set and validation set
    train_size  = 0.8
    inputs_train, inputs_valid = train_test_split(
        filenames, random_state=random_seed, train_size=train_size, shuffle=True
    )

    print("total: ", len(filenames), " train: ", len(inputs_train), " valid: ", len(inputs_valid))

    # Training dataset
    train_dataset = RegistrationDataSet(path_dataset = inputs_train,
                                        input_shape  = (300, 192, 192, 1),
                                        transform    = None)

    # Validation dataset
    valid_dataset = RegistrationDataSet(path_dataset = inputs_valid,
                                        input_shape  = (300, 192, 192, 1),
                                        transform    = None)

    # Training dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

    # Validation dataloader
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True)

    return train_dataloader, valid_dataloader


def save_table(table_name, fixed_img, moving_img, w0_img, w1_img, t1_img):
    table = wandb.Table(columns=['Fixed Image', 'Moving Image', 'Affine Reg. Image', 'Deformation Reg. Image', 'Deformation Field'], allow_mixed_types = True)
    
    saving_examples_folder = '/projects/disentanglement_methods/temp/PAM/results_thorax/images_PAM/'
    
    #PIL VERSION
    transform = T.ToPILImage()    
    fixed_img = transform(fixed_img[:,:,:,:,50].squeeze())
    moving_img = transform(moving_img[:,:,:,:,50].squeeze())
    affine_img = transform(w0_img[:,:,:,:,50].squeeze())
    deformation_img = transform(w1_img[:,:,:,:,50].squeeze())
    deformation_field = transform(t1_img[:,:,:,:,50].squeeze())

    fixed_img.show()                              
    fixed_img.save(saving_examples_folder + "fixed_image.jpg")    
    moving_img.show() 
    moving_img.save(saving_examples_folder + "moving_image.jpg")    
    affine_img.show() 
    affine_img.save(saving_examples_folder + "affine_image.jpg")    
    deformation_img.show()
    deformation_img.save(saving_examples_folder + "deformation_image.jpg")    
    deformation_field.show()
    deformation_field.save(saving_examples_folder + "deformation_field.jpg")  
    
    table.add_data(
        wandb.Image(Image.open(saving_examples_folder + "fixed_image.jpg")),
        wandb.Image(Image.open(saving_examples_folder + "moving_image.jpg")),
        wandb.Image(Image.open(saving_examples_folder + "affine_image.jpg")),
        wandb.Image(Image.open(saving_examples_folder + "deformation_image.jpg")),
        wandb.Image(Image.open(saving_examples_folder + "deformation_field.jpg"))
    )
    
    wandb.log({table_name: table})
    

def training(pam_net, dis_net, device, disc_loss, l2_loss, nn_loss, energy_loss, pam_optimizer, dis_optimizer,
             train_dataloader, valid_dataloader):
    print ('Starting training stage!')

    epoch        = 0
    n_epochs     = 10001
    train_losses = []
    valid_losses = []
    alpha_value  = 0.01
    beta_value   = 0.01
    gamma_value  = 0.1

    # Establish convention for real and fake labels during training
    real_label   = 1.
    fake_label   = 0.

    # wandb Initialization
    wandb.init(project='Adversarial-PAM', entity='ljestaciocerquin')

    # Saving model inputs and hyperparameters
    config          = wandb.config
    wandb.watch(pam_net, log='all')

    it_train_counter = 0
    it_valid_counter = 0
    train_flag = True
    
    fixed_draw = None
    moving_draw = None
    w_0_draw    = None
    w_1_draw   = None
    deform_draw = None


    for epoch in range(epoch, n_epochs):
       
        loss_pam_train    = 0
        loss_disc_train   = 0
        loss_pam_valid    = 0
        loss_disc_valid   = 0

        pam_net.train()
        dis_net.train()

        for i, (x_1, x_2) in enumerate(train_dataloader):

            train_flag = True

            # send to device (GPU or CPU)
            fixed  = x_1.to(device)
            moving = x_2.to(device)

            # -----------------
            #  Train Generator
            # -----------------

            # zero-grad the parameters
            pam_optimizer.zero_grad()

            # Forward pass generator (1)
            t_0, w_0, t_1, w_1 = pam_net(fixed, moving)

            # Compute distance between feature maps

            _, features_w1 = dis_net(w_1) 
            _, features_w0 = dis_net(w_0) 
            generator_adv_loss = l2_loss(features_w1, features_w0)

            # Generator: Affine network loss
            registration_affine_loss = nn_loss.pearson_correlation(fixed, w_0)
            penalty_affine_loss      = energy_loss.energy_loss(t_0)

            # Generator: Deformation network loss
            registration_deform_loss = nn_loss.pearson_correlation(fixed, w_1)
            penalty_deform_loss = energy_loss.energy_loss(t_1)
            
            # PAM loss
            loss = registration_affine_loss + alpha_value * penalty_affine_loss + registration_deform_loss + beta_value * penalty_deform_loss + gamma_value * generator_adv_loss
            loss_pam_train += loss.item()

            loss.backward()

            # update the parameters
            pam_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Backward pass discriminator
            dis_optimizer.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real, _ = dis_net(w_0.detach()) 
            fake, _ = dis_net(w_1.detach())

            b_size   = real.shape
            label_r  = torch.full(b_size, real_label, dtype=torch.float, device=device)
            label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=device)

            # Compute loss
            loss_d_real = disc_loss(real, label_r)
            loss_d_fake = disc_loss(fake, label_f)
            loss_d_t    = (loss_d_real + loss_d_fake) * 0.5
            loss_disc_train += loss_d_t.item()

            # one backward pass
            loss_d_t.backward()

            # Update Discriminator
            dis_optimizer.step()

            # Reinit the affine network weights
            if loss_d_t.item() < 1e-5:  # >
                dis_net.apply(weights_init)
                print("Reloading discriminator weights")


            # Display in tensorboard
            # ========
            wandb.log({'Iteration': it_train_counter, 
                       'Train: Similarity Affine loss': registration_affine_loss.item(),
                       'Train: Penalty Affine loss': alpha_value * penalty_affine_loss.item(),
                       'Train: Similarity Elastic loss': registration_deform_loss.item(),
                       'Train: Penalty Elastic loss': beta_value * penalty_deform_loss.item(),
                       'Train: Generator Adversarial Loss': generator_adv_loss.item(),
                       'Train: Total loss': loss.item(),
                       'Train: Discriminator Loss': loss_d_t.item()})

            if train_flag:
                it_train_counter += 1


        with torch.no_grad():

            train_flag = False

            pam_net.eval()
            dis_net.eval()

            for i, (x_1, x_2) in enumerate(valid_dataloader):
                # send to device (GPU or CPU)
                fixed  = x_1.to(device)
                moving = x_2.to(device)

                # Forward pass generator (1)
                t_0, w_0, t_1, w_1 = pam_net(fixed, moving)

                # Loss measures generator's ability to fool the discriminator
                _, features_w1 = dis_net(w_1) 
                _, features_w0 = dis_net(w_0) 
                generator_adv_loss = l2_loss(features_w1, features_w0)

                # Computing the Generator Loss
                # Affine network loss
                registration_affine_loss = nn_loss.pearson_correlation(fixed, w_0)
                penalty_affine_loss      = energy_loss.energy_loss(t_0)

                # Deformation network loss
                registration_deform_loss = nn_loss.pearson_correlation(fixed, w_1)
                penalty_deform_loss = energy_loss.energy_loss(t_1)

                # PAM loss
                # total loss
                loss = registration_affine_loss + alpha_value * penalty_affine_loss + registration_deform_loss + beta_value * penalty_deform_loss + gamma_value * generator_adv_loss
                loss_pam_valid += loss.item()


                # ----------- 1. Update the Discriminator -----------

                # Measure discriminator's ability to classify real from generated samples
                real, _ = dis_net(w_0)  # (fixed)
                fake, _ = dis_net(w_1.detach())

                b_size = real.shape
                label_r  = torch.full(b_size, real_label, dtype=torch.float, device=device)
                label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=device)

                # Calculate loss
                loss_d_real      = disc_loss(real, label_r)
                loss_d_fake      = disc_loss(fake, label_f)
                loss_d_v         = (loss_d_real + loss_d_fake) * 0.5
                loss_disc_valid += loss_d_v.item()


                # Display in weights and biases
                # ========
                wandb.log({'Iteration': it_valid_counter, 
                       'Valid: Similarity Affine loss': registration_affine_loss.item(),
                       'Valid: Penalty Affine loss': alpha_value * penalty_affine_loss.item(),
                       'Valid: Similarity Elastic loss': registration_deform_loss.item(),
                       'Valid: Penalty Elastic loss': beta_value * penalty_deform_loss.item(),
                       'Valid: Generator Adversarial Loss': generator_adv_loss.item(),
                       'Valid: Total loss': loss.item(),
                       'Valid: Discriminator Loss': loss_d_t.item()})

                if not train_flag:
                    it_valid_counter += 1
                
                fixed_draw = fixed
                moving_draw = moving
                w_0_draw    = w_0
                w_1_draw    = w_1
                deform_draw = t_1

        # Visualization of images
        save_table('Validation Images' ,fixed_draw, moving_draw, w_0_draw, w_1_draw, deform_draw)
        
        # Compute the loss per epoch
        loss_pam_train     /= len(train_dataloader)
        loss_disc_train    /= len(train_dataloader)

        # Save checkpoints
        if epoch % 10 == 0:
            name_pam = 'PAMModel_' + str(epoch) + '.pth'
            name_dis = 'DisModel_' + str(epoch) + '.pth'
            torch.save(pam_net.state_dict(), os.path.join(args_pam_adv_fts_sit.checkpoints_folder, name_pam))
            torch.save(dis_net.state_dict(), os.path.join(args_pam_adv_fts_sit.checkpoints_folder, name_dis))
            print('Saving model')

        # Compute the loss per epoch
        loss_pam_valid     /= len(valid_dataloader)
        loss_disc_valid    /= len(valid_dataloader)

        wandb.log({'epoch': epoch+1,
                    'Train: Total loss by epoch': loss_pam_train,
                    'Valid: Total loss by epoch': loss_pam_valid,
                    'Train: Discriminator Loss by epoch': loss_disc_train,
                    'Valid: Discriminator Loss by epoch': loss_disc_valid
                    })
        
        # Print the train and validation losses
        print("Train epoch : {}/{}, loss_PAM = {:.6f},".format(epoch, n_epochs, loss_pam_train)) 
        print("Train epoch : {}/{}, loss_Dis = {:.6f},".format(epoch, n_epochs, loss_disc_train))
        print("Valid epoch : {}/{}, loss_PAM = {:.6f},".format(epoch, n_epochs, loss_pam_valid))
        print("Valid epoch : {}/{}, loss_Dis = {:.6f},".format(epoch, n_epochs, loss_disc_valid))



def start_training():
    make_directory_to_save_chkp()
    cuda_seeds()
    pam_net, dis_net, device                 = model_init()
    disc_loss, l2_loss, nn_loss, energy_loss = init_loss_functions()
    pam_optimizer, dis_optimizer             = get_optimizers(pam_net, dis_net)
    train_dataloader, valid_dataloader       = load_dataloader()
    training(pam_net, dis_net, device, disc_loss, l2_loss, nn_loss, energy_loss, pam_optimizer, dis_optimizer,
             train_dataloader, valid_dataloader)


def load_model_weights():
    # Network definition
    pam_net     = PAMNetwork()
    dis_net     = DiscriminatorNetwork()

    # GPU computation
    device      = torch.device('cuda:0')
    pam_net.to(device)
    dis_net.to(device)

    # Loading the model weights
    pam_chkpt = args_pam_adv_fts_sit.pam_checkpoint
    dis_chkpt = args_pam_adv_fts_sit.dis_checkpoint
    pam_net.load_state_dict(torch.load(pam_chkpt))
    dis_net.load_state_dict(torch.load(dis_chkpt))

    return pam_net, dis_net, device


def start_retraining():
    make_directory_to_save_chkp()
    cuda_seeds()
    pam_net, dis_net, device           = load_model_weights() 
    disc_loss, l2_loss, nn_loss, energy_loss = init_loss_functions()
    pam_optimizer, dis_optimizer       = get_optimizers(pam_net, dis_net)
    train_dataloader, valid_dataloader = load_dataloader()
    training(pam_net, dis_net, device,disc_loss, l2_loss, nn_loss, energy_loss, pam_optimizer, dis_optimizer,
             train_dataloader, valid_dataloader)

start_training()
#start_retraining()
print("End Training :)")