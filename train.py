import os
import torch
import wandb
from   utils                        import create_directory
from   utils                        import cuda_seeds
from   utils                        import weights_init
from   utils                        import read_3D_survival_train_valid_data
from   utils                        import save_images_weights_and_biases
from   losses                       import Cross_Correlation_Loss
from   losses                       import Energy_Loss
from   registration_dataset         import Registration2DDataSet
from   registration_dataset         import Registration3DDataSet
from   networks.registration_model  import Registration_PAM
from   networks.registration_model  import Registration_PAM_Survival
from   networks.discriminator       import Discriminator
from   torch.utils.data             import DataLoader
from   sklearn.model_selection      import train_test_split
import matplotlib.pyplot            as     plt
import numpy                        as     np
import torchvision.transforms.functional as TF
from random import randint

class Train(object):
    def __init__(self, args):

        self.input_ch   = args.input_ch
        self.input_dim  = args.input_dim
        self.latent_dim = args.latent_dim
        self.output_ch  = args.output_ch
        self.group_num  = args.group_num
        self.filters    = args.filters
        self.filters_discriminator  = args.filters_disc
        self.input_ch_discriminator = args.input_ch_d
        
        # Train split
        self.train_split = args.train_split
        
        # Model
        self.add_survival = args.add_survival
        
        # Model Parameters
        self.lr        = args.lr
        self.beta1     = args.beta1
        self.beta2     = args.beta2
        self.batch_size= args.batch_size
        self.mse_loss  = torch.nn.MSELoss()
        
        # Path to save checkpoints and results
        self.checkpoints_folder = args.ckpt_dir
        self.results_dir        = args.results_dir
        
        # Device
        self.num_gpus = args.num_gpus
        self.device   = torch.device("cuda:0" if (torch.cuda.is_available() and self.num_gpus > 0) else "cpu")
        
        # Values of the regularization parameters
        self.alpha_value  = args.alpha_value            # regularization for the affine penalty loss
        self.beta_value   = args.beta_value             # regularization for the deformation penalty loss
        self.gamma_value  = args.gamma_value            # regularization for the discriminator (feature matching loss: MSE)
    
        # Data folder
        self.data_file = args.dataset_file
        
        # Number of epochs to train the model
        self.n_epochs    = args.n_epochs
        self.start_epoch = args.start_ep #1
        
        # Variables to save weights and biases images
        self.fixed_draw  = None
        self.moving_draw = None
        self.w_0_draw    = None
        self.w_1_draw    = None
        
        # Directory to save checkpoints
        create_directory(self.checkpoints_folder)
        create_directory(self.results_dir)
        
        # Cuda Seeds for reproducibility
        cuda_seeds()
        
    
    def model_init(self):
        if self.add_survival:
            net = Registration_PAM_Survival
        else:
            net = Registration_PAM
        
        # Network Definition
        self.pam_net = net( input_ch   = self.input_ch,
                        input_dim  = self.input_dim,
                        latent_dim = self.latent_dim,
                        output_ch  = self.output_ch,
                        group_num  = self.group_num,
                        filters    = self.filters)
        
        self.discriminator_net  = Discriminator(
                input_ch   = self.input_ch_discriminator,
                input_dim  = self.input_dim,
                group_num  = self.group_num,
                filters    = self.filters) 
        
        # GPU computation
        self.pam_net.to(self.device)
        self.discriminator_net.to(self.device)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.num_gpus > 1):
            self.pam_net = torch.nn.DataParallel(self.pam_net, list(range(self.num_gpus)))
            self.discriminator_net = torch.nn.DataParallel(self.discriminator_net, list(range(self.num_gpus)))
        
        # Init weights for the model
        self.pam_net.apply(weights_init)
        self.discriminator_net.apply(weights_init)
    
    
    def set_losses(self):
        # Loss definitions
        self.nn_loss       = Cross_Correlation_Loss()
        self.energy_loss   = Energy_Loss()
        self.disc_loss     = torch.nn.BCELoss()
        self.l2_loss       = torch.nn.MSELoss()
        self.survival_loss = torch.nn.BCELoss()

        
    def set_optimizer(self):
        self.pam_optim  = torch.optim.Adam(self.pam_net.parameters(), lr=self.lr,
                                           betas=(self.beta1, self.beta2))
        self.disc_optim = torch.optim.Adam(self.discriminator_net.parameters(), lr=self.lr, 
                                           betas=(self.beta1, self.beta2))
        
                    
    def load_dataloader(self):
        # Dataset Path 
        inputs_train, inputs_valid = read_3D_survival_train_valid_data(self.data_file)
        '''# Random seed
        random_seed = 42

        # Split dataset into training set and validation set
        inputs_train, inputs_valid = train_test_split(
            filenames, random_state=random_seed, train_size=self.train_split, shuffle=True, stratify=filenames['PatientID']
        )

        print("total: ", len(filenames), " train: ", len(inputs_train), " valid: ", len(inputs_valid))'''
        
        if len(self.input_dim) == 2:
            registration_dataset = Registration2DDataSet
        else:
            registration_dataset = Registration3DDataSet

        # Training and Validation dataset
        train_dataset = registration_dataset(path_dataset = inputs_train,
                                             input_dim  = self.input_dim, # [160, 192, 192] -> (160, 192, 192, 1)
                                             transform    = None)

        valid_dataset = registration_dataset(path_dataset = inputs_valid,
                                             input_dim  = self.input_dim, # [160, 192, 192] -> (160, 192, 192, 1)
                                             transform    = None)

        # Training and Validation dataloader
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=True)
    

    def train_PAM_Adversarial(self):
        
        # weights and biases
        if self.add_survival:
            project_name = 'Adversarial_PAM_Survival'
        else:
            project_name = 'Adversarial_PAM'
            
        wandb.init(project=project_name, entity='ljestaciocerquin')
        config = wandb.config
        wandb.watch(self.pam_net, log="all")
                      
        # Establish convention for real and fake labels during training
        real_label   = 1.
        fake_label   = 0.
        
        for epoch in range(self.start_epoch, self.n_epochs):

            # Total loss train
            loss_pam_train    = 0
            loss_disc_train   = 0
            loss_pam_valid    = 0
            loss_disc_valid   = 0
            
            # Set the training mode
            self.pam_net.train()
            self.discriminator_net.train()
            

            for i, (x_1, x_2) in enumerate (self.train_dataloader):
                
                # send to device (GPU or CPU)
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
                
                # -----------------
                #  Train Generator
                # -----------------
                
                # zero-grad the net parameters
                self.pam_optim.zero_grad()
                
                # Forward pass generator (1)
                t_0, w_0, t_1, w_1 = self.pam_net(fixed, moving)
                
                # Compute distance between feature maps
                _, features_w1     = self.discriminator_net(w_1)
                _, features_w0     = self.discriminator_net(w_0)
                generator_adv_loss = self.l2_loss(features_w1, features_w0)
                
                # Generator: Affine network loss
                registration_affine_loss = self.nn_loss.pearson_correlation(fixed, w_0)
                penalty_affine_loss      = self.energy_loss.energy_loss(t_0)
                
                # Generator: Deformation network loss
                registration_deform_loss = self.nn_loss.pearson_correlation(fixed, w_1)
                penalty_deform_loss      = self.energy_loss.energy_loss(t_1)

                # PAM loss
                loss = registration_affine_loss + self.alpha_value * penalty_affine_loss +\
                       registration_deform_loss + self.beta_value  * penalty_deform_loss +\
                       self.gamma_value * generator_adv_loss
                loss_pam_train += loss.item()
                
                # one backward pass
                loss.backward()
                
                # Update the parameters
                self.pam_optim.step()
                
                
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Backward pass discriminator
                self.disc_optim.zero_grad()
                
                # Measure discriminator's ability to classify real from generated samples
                real, _  = self.discriminator_net(w_0.detach())
                fake, _  = self.discriminator_net(w_1.detach())
                b_size   = real.shape
                label_r  = torch.full(b_size, real_label, dtype=torch.float, device=self.device)
                label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=self.device)
                
                # Compute discriminator loss
                loss_d_real = self.disc_loss(real, label_r)
                loss_d_fake = self.disc_loss(fake, label_f)
                loss_discriminator = (loss_d_real + loss_d_fake) * 0.5
                loss_disc_train   += loss_discriminator.item()
                
                # one backward pass
                loss_discriminator.backward()

                # Update Discriminator
                self.disc_optim.step()
                
                # Reinit the affine network weights
                if loss_discriminator.item() < 1e-5:  # 
                    self.discriminator_net.apply(weights_init)
                    print("Reloading discriminator weights")
                
                it_train_counter = len(self.train_dataloader)
                wandb.log({'Iteration': epoch * it_train_counter + i,
                            'Train: Similarity Affine loss': registration_affine_loss.item(),
                            'Train: Penalty Affine loss': self.alpha_value * penalty_affine_loss.item(),
                            'Train: Similarity Elastic loss': registration_deform_loss.item(),
                            'Train: Penalty Elastic loss': self.beta_value * penalty_deform_loss.item(),
                            'Train: Generator Adversarial Loss': self.gamma_value * generator_adv_loss.item(),
                            'Train: Total loss': loss.item(),
                            'Train: Discriminator Loss': loss_discriminator.item()
                        })
                
                
            
            with torch.no_grad():
                self.pam_net.eval()
                self.discriminator_net.eval()
                
                for i, (x_1, x_2) in enumerate (self.valid_dataloader):
                    # send to device (GPU or CPU)
                    fixed  = x_1.to(self.device)
                    moving = x_2.to(self.device)
                    
                    # Forward pass generator (1)
                    t_0, w_0, t_1, w_1 = self.pam_net(fixed, moving)
                    
                    # Loss measures generator's ability to fool the discriminator. Compute distance between feature maps
                    _, features_w1     = self.discriminator_net(w_1)
                    _, features_w0     = self.discriminator_net(w_0)
                    generator_adv_loss = self.l2_loss(features_w1, features_w0)
                    
                    # Generator: Affine network loss
                    registration_affine_loss = self.nn_loss.pearson_correlation(fixed, w_0)
                    penalty_affine_loss      = self.energy_loss.energy_loss(t_0)
                    
                    # Generator: Deformation network loss
                    registration_deform_loss = self.nn_loss.pearson_correlation(fixed, w_1)
                    penalty_deform_loss      = self.energy_loss.energy_loss(t_1)

                    # PAM loss
                    loss = registration_affine_loss + self.alpha_value * penalty_affine_loss +\
                           registration_deform_loss + self.beta_value  * penalty_deform_loss +\
                           self.gamma_value * generator_adv_loss
                    loss_pam_valid += loss.item()
                    
                    
                    # ---------------------
                    #  Update the Discriminator
                    # ---------------------

                    # Measure discriminator's ability to classify real from generated samples
                    real, _  = self.discriminator_net(w_0.detach())
                    fake, _  = self.discriminator_net(w_1.detach())
                    b_size   = real.shape
                    label_r  = torch.full(b_size, real_label, dtype=torch.float, device=self.device)
                    label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=self.device)
                    
                    # Compute discriminator loss
                    loss_d_real = self.disc_loss(real, label_r)
                    loss_d_fake = self.disc_loss(fake, label_f)
                    loss_discriminator = (loss_d_real + loss_d_fake) * 0.5
                    loss_disc_valid   += loss_discriminator.item()
                    
                    it_valid_counter = len(self.it_valid_counter)
                    wandb.log({'Valid: Similarity Affine loss': registration_affine_loss.item(),
                               'Valid: Penalty Affine loss': self.alpha_value * penalty_affine_loss.item(),
                               'Valid: Similarity Elastic loss': registration_deform_loss.item(),
                               'Valid: Penalty Elastic loss': self.beta_value * penalty_deform_loss.item(),
                               'Valid: Generator Adversarial Loss': self.gamma_value * generator_adv_loss.item(),
                               'Valid: Total loss': loss.item(),
                               'Valid: Discriminator Loss': loss_discriminator.item()
                            })
                    
                    self.fixed_draw = fixed
                    self.moving_draw = moving
                    self.w_0_draw    = w_0
                    self.w_1_draw    = w_1

            # Visualization of images
            save_images_weights_and_biases('Validation Images', self.results_dir, self.fixed_draw, self.moving_draw, self.w_0_draw, self.w_1_draw)
        
            # Save checkpoints
            if epoch % 10 == 0:
                name_pam = 'PAM_Model_' + str(epoch) + '.pth'
                name_dis = 'Dis_Model_' + str(epoch) + '.pth'
                torch.save(self.pam_net.state_dict(), os.path.join(self.checkpoints_folder, name_pam))
                torch.save(self.discriminator_net.state_dict(), os.path.join(self.checkpoints_folder, name_dis))
                print('Saving model')

            # Train loss per epoch
            loss_pam_train  /= len(self.train_dataloader)
            loss_disc_train /= len(self.train_dataloader)

            # Valid loss per epoch
            loss_pam_valid  /= len(self.valid_dataloader)
            loss_disc_valid /= len(self.valid_dataloader)
            
            wandb.log({'epoch': epoch,
                    'Train: Total loss by epoch': loss_pam_train,
                    'Valid: Total loss by epoch': loss_pam_valid,
                    'Train: Discriminator Loss by epoch': loss_disc_train,
                    'Valid: Discriminator Loss by epoch': loss_disc_valid
                    })
            
        
            # Print the train and validation losses
            print("Train epoch : {}/{}, loss_PAM = {:.6f},".format(epoch, self.n_epochs, loss_pam_train)) 
            print("Train epoch : {}/{}, loss_Dis = {:.6f},".format(epoch, self.n_epochs, loss_disc_train))
            print("Valid epoch : {}/{}, loss_PAM = {:.6f},".format(epoch, self.n_epochs, loss_pam_valid))
            print("Valid epoch : {}/{}, loss_Dis = {:.6f},".format(epoch, self.n_epochs, loss_disc_valid))
    
    
    
    def train_PAM_Adversarial_Survival(self):
        
        # weights and biases
        if self.add_survival:
            project_name = 'Adversarial_PAM_Survival'
        else:
            project_name = 'Adversarial_PAM'
            
        wandb.init(project=project_name, entity='ljestaciocerquin')
        config = wandb.config
        wandb.watch(self.pam_net, log="all")
                      
        # Establish convention for real and fake labels during training
        real_label   = 1.
        fake_label   = 0.
        
        for epoch in range(self.start_epoch, self.n_epochs):

            # Total loss train
            loss_pam_train  = 0
            loss_disc_train = 0
            loss_surv_train = 0
            loss_pam_valid  = 0
            loss_disc_valid = 0
            loss_surv_valid = 0
            
            # Set the training mode
            self.pam_net.train()
            self.discriminator_net.train()
            

            for i, (x_1, x_2, surv) in enumerate (self.train_dataloader):
                
                # send to device (GPU or CPU)
                fixed       = x_1.to(self.device)
                moving      = x_2.to(self.device)
                surv_target = surv.to(self.device)
                
                # -----------------
                #  Train Generator
                # -----------------
                
                # zero-grad the net parameters
                self.pam_optim.zero_grad()
                
                # Forward pass generator (1)
                t_0, w_0, t_1, w_1, surv_pred = self.pam_net(fixed, moving)
                
                # Compute distance between feature maps
                _, features_w1     = self.discriminator_net(w_1)
                _, features_w0     = self.discriminator_net(w_0)
                generator_adv_loss = self.l2_loss(features_w1, features_w0)
                
                # Generator: Affine network loss
                registration_affine_loss = self.nn_loss.pearson_correlation(fixed, w_0)
                penalty_affine_loss      = self.energy_loss.energy_loss(t_0)
                
                # Generator: Deformation network loss
                registration_deform_loss = self.nn_loss.pearson_correlation(fixed, w_1)
                penalty_deform_loss      = self.energy_loss.energy_loss(t_1)
                
                # Generator: Survival Loss
                print(surv_pred.shape, surv_target.shape, ' ************************* ')
                survival_loss    = self.survival_loss(surv_pred, surv_target)
                loss_surv_train += survival_loss.item()
                
                # PAM loss
                loss = registration_affine_loss + self.alpha_value * penalty_affine_loss +\
                       registration_deform_loss + self.beta_value  * penalty_deform_loss +\
                       self.gamma_value * generator_adv_loss + survival_loss
                loss_pam_train += loss.item()
                
                # one backward pass
                loss.backward()
                
                # Update the parameters
                self.pam_optim.step()
                
                
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Backward pass discriminator
                self.disc_optim.zero_grad()
                
                # Measure discriminator's ability to classify real from generated samples
                real, _  = self.discriminator_net(w_0.detach()) 
                fake, _  = self.discriminator_net(w_1.detach())
                b_size   = real.shape
                label_r  = torch.full(b_size, real_label, dtype=torch.float, device=self.device)
                label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=self.device)
                
                # Compute discriminator loss
                loss_d_real = self.disc_loss(real, label_r)
                loss_d_fake = self.disc_loss(fake, label_f)
                loss_discriminator = (loss_d_real + loss_d_fake) * 0.5
                loss_disc_train   += loss_discriminator.item()
                
                # one backward pass
                loss_discriminator.backward()

                # Update Discriminator
                self.disc_optim.step()
                
                # Reinit the affine network weights
                if loss_discriminator.item() < 1e-5:  # 
                    self.discriminator_net.apply(weights_init)
                    print("Reloading discriminator weights")
                
                it_train_counter = len(self.train_dataloader)
                wandb.log({'Iteration': epoch * it_train_counter + i,
                            'Train: Similarity Affine loss': registration_affine_loss.item(),
                            'Train: Penalty Affine loss': self.alpha_value * penalty_affine_loss.item(),
                            'Train: Similarity Elastic loss': registration_deform_loss.item(),
                            'Train: Penalty Elastic loss': self.beta_value * penalty_deform_loss.item(),
                            'Train: Generator Adversarial Loss': self.gamma_value * generator_adv_loss.item(),
                            'Train: Survival Loss': survival_loss.item(),
                            'Train: Total loss': loss.item(),
                            'Train: Discriminator Loss': loss_discriminator.item()
                        })
                
                
            
            with torch.no_grad():
                self.pam_net.eval()
                self.discriminator_net.eval()
                
                for i, (x_1, x_2, surv) in enumerate (self.valid_dataloader):
                    # send to device (GPU or CPU)
                    fixed       = x_1.to(self.device)
                    moving      = x_2.to(self.device)
                    surv_target = surv.to(self.device)
                    
                    # Forward pass generator (1)
                    t_0, w_0, t_1, w_1, surv_pred = self.pam_net(fixed, moving)
                    
                    # Loss measures generator's ability to fool the discriminator. Compute distance between feature maps
                    _, features_w1     = self.discriminator_net(w_1)
                    _, features_w0     = self.discriminator_net(w_0)
                    generator_adv_loss = self.l2_loss(features_w1, features_w0)
                    
                    # Generator: Affine network loss
                    registration_affine_loss = self.nn_loss.pearson_correlation(fixed, w_0)
                    penalty_affine_loss      = self.energy_loss.energy_loss(t_0)
                    
                    # Generator: Deformation network loss
                    registration_deform_loss = self.nn_loss.pearson_correlation(fixed, w_1)
                    penalty_deform_loss      = self.energy_loss.energy_loss(t_1)
                    
                    # Generator: Survival Loss
                    survival_loss = self.survival_loss(surv_pred, surv_target)
                    loss_surv_valid += survival_loss.item()

                    # PAM loss
                    loss = registration_affine_loss + self.alpha_value * penalty_affine_loss +\
                           registration_deform_loss + self.beta_value  * penalty_deform_loss +\
                           self.gamma_value * generator_adv_loss + survival_loss
                    loss_pam_valid += loss.item()
                    
                    
                    
                    # ---------------------
                    #  Update the Discriminator
                    # ---------------------

                    # Measure discriminator's ability to classify real from generated samples
                    real, _  = self.discriminator_net(w_0.detach())
                    fake, _  = self.discriminator_net(w_1.detach())
                    b_size   = real.shape
                    label_r  = torch.full(b_size, real_label, dtype=torch.float, device=self.device)
                    label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=self.device)
                    
                    # Compute discriminator loss
                    loss_d_real = self.disc_loss(real, label_r)
                    loss_d_fake = self.disc_loss(fake, label_f)
                    loss_discriminator = (loss_d_real + loss_d_fake) * 0.5
                    loss_disc_valid   += loss_discriminator.item()
                    
                    it_valid_counter = len(self.it_valid_counter)
                    wandb.log({'Valid: Similarity Affine loss': registration_affine_loss.item(),
                               'Valid: Penalty Affine loss': self.alpha_value * penalty_affine_loss.item(),
                               'Valid: Similarity Elastic loss': registration_deform_loss.item(),
                               'Valid: Penalty Elastic loss': self.beta_value * penalty_deform_loss.item(),
                               'Valid: Generator Adversarial Loss': self.gamma_value * generator_adv_loss.item(),
                               'Valid: Survival Loss': survival_loss.item(),
                               'Valid: Total loss': loss.item(),
                               'Valid: Discriminator Loss': loss_discriminator.item()
                            })
                    
                    self.fixed_draw = fixed
                    self.moving_draw = moving
                    self.w_0_draw    = w_0
                    self.w_1_draw    = w_1

            # Visualization of images
            save_images_weights_and_biases('Validation Images', self.results_dir, self.fixed_draw, self.moving_draw, self.w_0_draw, self.w_1_draw)
        
            # Save checkpoints
            if epoch % 10 == 0:
                name_pam = 'PAM_Model_' + str(epoch) + '.pth'
                name_dis = 'Dis_Model_' + str(epoch) + '.pth'
                torch.save(self.pam_net.state_dict(), os.path.join(self.checkpoints_folder, name_pam))
                torch.save(self.discriminator_net.state_dict(), os.path.join(self.checkpoints_folder, name_dis))
                print('Saving model')

            # Train loss per epoch
            loss_pam_train  /= len(self.train_dataloader)
            loss_disc_train /= len(self.train_dataloader)
            loss_surv_train /= len(self.train_dataloader)

            # Valid loss per epoch
            loss_pam_valid  /= len(self.valid_dataloader)
            loss_disc_valid /= len(self.valid_dataloader)
            loss_surv_valid /= len(self.valid_dataloader)
            
            wandb.log({'epoch': epoch,
                    'Train: Total loss by epoch': loss_pam_train,
                    'Valid: Total loss by epoch': loss_pam_valid,
                    'Train: Survival loss by epoch': loss_surv_train,
                    'Valid: Survival loss by epoch': loss_surv_valid,
                    'Train: Discriminator Loss by epoch': loss_disc_train,
                    'Valid: Discriminator Loss by epoch': loss_disc_valid
                    })
            
        
            # Print the train and validation losses
            print("Train epoch : {}/{}, loss_PAM = {:.6f},".format(epoch, self.n_epochs, loss_pam_train)) 
            print("Train epoch : {}/{}, loss_Sur = {:.6f},".format(epoch, self.n_epochs, loss_surv_train)) 
            print("Train epoch : {}/{}, loss_Dis = {:.6f},".format(epoch, self.n_epochs, loss_disc_train))
            print("Valid epoch : {}/{}, loss_PAM = {:.6f},".format(epoch, self.n_epochs, loss_pam_valid))
            print("Valid epoch : {}/{}, loss_Sur = {:.6f},".format(epoch, self.n_epochs, loss_surv_valid)) 
            print("Valid epoch : {}/{}, loss_Dis = {:.6f},".format(epoch, self.n_epochs, loss_disc_valid))
                
    
    def train_registration_method(self):
        self.model_init()
        self.set_losses()
        self.set_optimizer()
        self.load_dataloader()

        if self.add_survival:
            self.train_PAM_Adversarial_Survival()
            print('-------------- Running Adversarial PAM for Survival --------------')
        else:
            self.train_PAM_Adversarial()
            print('-------------- Running Adversarial PAM --------------')
        