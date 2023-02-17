import os
import torch
import wandb
from   utils                        import create_directory
from   utils                        import cuda_seeds
from   utils                        import weights_init
from   utils                        import read_train_data
from   losses                       import *
from   registration_dataset         import RegistrationDataSet
from   networks.registration_model  import Registration_Beta_VAE
from   networks.registration_model  import Registration_Wasserstein_AE
from   networks.discriminator       import Discriminator
from   torch.utils.data             import DataLoader
from   sklearn.model_selection      import train_test_split
import matplotlib.pyplot            as     plt
import numpy                        as     np
import torchvision.transforms.functional as TF
from random import randint

class Disentanglement(object):
    def __init__(self, args):

        self.input_ch   = args.input_ch
        self.input_dim  = args.data_dim
        self.latent_dim = args.z_dim
        self.output_ch  = args.output_ch
        self.group_num  = args.group_num
        self.filters    = [32, 64, 128, 256]
        self.input_ch_discriminator = args.input_ch_d
        
        # Model
        self.model        = args.model
        self.add_discriminator = args.add_disc
        
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
        
        # Beta value for Beta_VAE
        self.beta = 0
        
        # Data folder
        self.data_folder = args.dset_dir
        
        # Number of epochs to train the model
        self.n_epochs    = args.n_epochs
        self.start_epoch = args.start_ep #1
        
        # Directory to save checkpoints
        create_directory(self.checkpoints_folder)
        create_directory(self.results_dir)
        
        # Cuda Seeds for reproducibility
        cuda_seeds()
        
    
    def model_init(self):
        if self.model == 'WAE':
            net = Registration_Wasserstein_AE
            print('-------------- Running WAE Model --------------')
        elif self.model == 'Beta-VAE':
            net = Registration_Beta_VAE
            print('-------------- Running Beta-VAE Model --------------')
        else:
            raise NotImplementedError('only support model WAE and B-VAE')
        
        # Network Definition
        self.net = net( input_ch   = self.input_ch,
                        input_dim  = self.input_dim,
                        latent_dim = self.latent_dim,
                        output_ch  = self.output_ch,
                        group_num  = self.group_num,
                        filters    = self.filters)
        
        # GPU computation
        self.net.to(self.device)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.num_gpus > 1):
            self.net = torch.nn.DataParallel(self.net, list(range(self.num_gpus)))
        
        # Init weights for the model
        self.net.apply(weights_init)
        self.nn_loss     = Cross_Correlation_Loss()
        self.energy_loss = Energy_Loss()


        # Adversarial Learning
        if self.add_discriminator:
            self.discriminator_net  = Discriminator(
                input_ch   = self.input_ch_discriminator,
                data_dim   = self.input_dim,
                latent_dim = self.latent_dim,
                group_num  = self.group_num,
                filters    = self.filters)
            
            self.discriminator_net.to(self.device)
            # Handle multi-gpu if desired
            if (self.device.type == 'cuda') and (self.num_gpus > 1):
                self.discriminator_net = torch.nn.DataParallel(self.discriminator_net, list(range(self.num_gpus)))
            self.discriminator_net.apply(weights_init)
            self.disc_loss_bce = torch.nn.BCELoss()
            self.disc_loss_fts = torch.nn.MSELoss()
            print('-------------- Running Adversarial Beta-VAE Model --------------')
            
            
        
    def set_optimizer(self):
        self.optim      = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                                           betas=(self.beta1, self.beta2))
        self.optim_disc = torch.optim.Adam(self.discriminator_net.parameters(), lr=self.lr, 
                                           betas=(self.beta1, self.beta2))
        
                    
    def load_dataloader(self):
        # Dataset Path 
        filenames = read_train_data(self.data_folder)
       
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
                                            transform    = None)

        # Validation dataset
        valid_dataset = RegistrationDataSet(path_dataset = inputs_valid,
                                            transform    = None)

        # Training dataloader
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation dataloader
        self.valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=True)
    
    
           
    def train_Beta_VAE(self):
        
        # weights and biases
        wandb.init(project='Beta-VAE', entity='ljestaciocerquin')
        config = wandb.config
        wandb.watch(self.net, log="all")
        alpha_value = 0.01
        
        for epoch in range(self.start_epoch, self.n_epochs):
                        
            # Total loss 
            loss_pam_beta_vae_train= 0
            
            # Total loss 
            loss_pam_beta_vae_valid= 0
            lambda_value           = 0.001
            
            # Set the training mode
            self.net.train()
            
            if epoch >= 50: 
                self.beta = np.maximum(self.beta, 1e-10)
                self.beta = np.minimum(self.beta*2, 4.)      
                                        
            
            for i, (x_1, x_2) in enumerate (self.train_dataloader):
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
               
                # zero-grad the net parameters
                self.optim.zero_grad()
                
                # Forward pass through the registration model
                t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                
                # Computing the affine loss
                affine_mse_loss  = self.mse_loss(w_0, fixed)
                #registration_affine_cc_loss = nn_loss.pearson_correlation(fixed, w_0)
                #penalty_affine_loss = energy_loss.energy_loss(t_0)
                
                # Computing the elastic loss: Beta-VAE loss
                reconstruction_loss = lambda_value * reconstruction_loss(fixed, w_1) 
                kl_divergence_loss  = kl_divergence(mu, log_var)
                              
                # Total loss
                loss = affine_mse_loss + (reconstruction_loss + self.beta * kl_divergence_loss)
                loss_pam_beta_vae_train += loss.item()
                
                # one backward pass
                loss.backward()
                
                # Update the parameters
                self.optim.step()

                # Weights and biases visualization
                wandb.log({'Iteration': i,
                        'Train: Affine loss': affine_mse_loss.item(),
                        'Train: Reconstruction loss': reconstruction_loss.item(),
                        'Train: KL-divergence Loss': kl_divergence_loss.item(),
                        'Train: Beta-VAE Loss': reconstruction_loss.item() + self.beta * kl_divergence_loss.item(),
                        'Train: Total loss': loss.item()})
                
            
            with torch.no_grad():
                self.net.eval()
                
                for i, (x_1, x_2) in enumerate (self.valid_dataloader):
                    fixed  = x_1.to(self.device)
                    moving = x_2.to(self.device)
                                        
                    # Forward pass through the registration model
                    t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)

                    # Computing the affine loss
                    affine_mse_loss  = self.mse_loss(w_0, fixed)
                    #registration_affine_cc_loss = nn_loss.pearson_correlation(fixed, w_0)
                    #penalty_affine_loss = energy_loss.energy_loss(t_0)
                    
                    # Computing the elastic loss: Beta-VAE loss
                    reconstruction_loss = lambda_value * reconstruction_loss(fixed, w_1) 
                    kl_divergence_loss  = kl_divergence(mu, log_var)
                    
                                                   
                    # Computing the elastic loss: Beta-VAE loss
                    loss = affine_mse_loss + (reconstruction_loss + self.beta * kl_divergence_loss)
                    
                    # Total loss
                    loss_pam_beta_vae_valid += loss.item()

                    # Weights and biases visualization
                    wandb.log({'Iteration': i,
                            'Valid: Affine loss': affine_mse_loss.item(),
                            'Valid: Reconstruction loss': reconstruction_loss.item(),
                            'Valid: KL-divergence Loss': kl_divergence_loss.item(),
                            'Valid: Beta-VAE Loss': reconstruction_loss.item() + self.beta * kl_divergence_loss.item(),
                            'Valid: Total loss': loss.item()})
        
            # Save checkpoints
            if epoch % 10 == 0:
                name_pam = 'PAMModel_BetaVAE_' + str(epoch) + '.pth'
                torch.save(self.net.state_dict(), os.path.join(self.checkpoints_folder, name_pam))
                print('Saving model')

            # Compute the loss per epoch
            data_loader_len         = len(self.train_dataloader)
            loss_pam_beta_vae_train/= data_loader_len
            # Compute the loss per epoch
            data_loader_len         = len(self.valid_dataloader)
            loss_pam_beta_vae_valid/= data_loader_len
        
            # Print the train and validation losses
            print("Train epoch : {}/{}, loss_PAM = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_train, self.beta)) # epoch + 1, n_epochs
            print("Valid epoch : {}/{}, loss_PAM = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_valid, self.beta))



    def train_Beta_VAE_Adversarial(self):
        
        # weights and biases
        wandb.init(project='Beta-VAE-Adversarial', entity='ljestaciocerquin')
        config = wandb.config
        wandb.watch(self.net, log="all")
        
        # Assign lambda and gamma values to regularize the reconstruction 
        lambda_value = 0.001
        gamma_value  = 0.1
                
        # Establish convention for real and fake labels during training
        real_label   = 1.
        fake_label   = 0.
        
        for epoch in range(self.start_epoch, self.n_epochs):

            # Total loss train
            loss_pam_beta_vae_train= 0
            loss_disc_train        = 0
            
            # Total loss valid
            loss_pam_beta_vae_valid= 0
            loss_disc_valid        = 0
            
            # Set the training mode
            self.net.train()
            self.discriminator_net.train()
            
            # Update the beta value gradually
            if epoch >= 50: 
                self.beta = np.maximum(self.beta, 1e-10)
                self.beta = np.minimum(self.beta*2, 4.)  
            
            angle = randint(0, 20)    
                                        
            
            for i, (x_1, x_2) in enumerate (self.train_dataloader):
                
                # send to device (GPU or CPU)
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
                
                # -----------------
                #  Train Generator
                # -----------------
                
                # zero-grad the net parameters
                self.optim.zero_grad()
                
                # Forward pass through the registration model: generator (1)
                t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                
                # Loss measures generator's ability to fool the discriminator
                _, features_w1    = self.discriminator_net(w_1) # features_w0
                _, features_fixed = self.discriminator_net(TF.rotate(fixed, angle)) 
                
                # Compute generator loss
                generator_mse_penalty  = gamma_value * self.disc_loss_fts(features_w1, features_fixed)
                
                # Computing the affine loss
                affine_mse_loss    = self.mse_loss(w_0, fixed)
                
                # Computing the elastic loss: Beta-VAE loss
                reconstruction_loss = lambda_value * reconstruction_loss(fixed, w_1) 
                kl_divergence_loss  = kl_divergence(mu, log_var)

                # Total loss Beta-VAE train
                loss_generator = affine_mse_loss + (reconstruction_loss + self.beta * kl_divergence_loss) + generator_mse_penalty
                loss_pam_beta_vae_train += loss_generator.item()
                
                # one backward pass
                loss_generator.backward()
                
                # Update the parameters
                self.optim.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Backward pass discriminator
                self.optim_disc.zero_grad()
                
                # Measure discriminator's ability to classify real from generated samples
                real, _  = self.discriminator_net(TF.rotate(fixed, angle))  # Shall we compare the features as well?
                fake, _  = self.discriminator_net(w_1.detach())
                b_size   = real.shape
                label_r  = torch.full(b_size, real_label, dtype=torch.float, device=self.device)
                label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=self.device)
                
                # Compute discriminator loss
                loss_d_real = self.disc_loss_bce(real, label_r)
                loss_d_fake = self.disc_loss_bce(fake, label_f)
                loss_discriminator    = (loss_d_real + loss_d_fake) * 0.5
                loss_disc_train += loss_discriminator.item()
                
                # one backward pass
                loss_discriminator.backward()

                # Update Discriminator
                self.optim_disc.step()
                
                # Reinit the affine network weights
                if loss_discriminator.item() < 1e-5:  # 
                    self.discriminator_net.apply(weights_init)
                    print("Reloading discriminator weights")
                
                wandb.log({'epoch': i,
                            'Train: Affine loss': affine_mse_loss.item(),
                            'Train: Reconstruction loss': reconstruction_loss.item(),
                            'Train: KL-divergence Loss': kl_divergence_loss.item(),
                            'Train: Beta-VAE Loss': reconstruction_loss.item() + self.beta * kl_divergence_loss.item(),
                            'Train: FTS Generator Loss': generator_mse_penalty.item(),
                            'Train: Generator Total loss': loss_generator.item(),
                            'Train: Discriminator loss': loss_discriminator,
                            })
                
                
            
            with torch.no_grad():
                self.net.eval()
                self.discriminator_net.eval()
                
                for i, (x_1, x_2) in enumerate (self.valid_dataloader):
                    fixed  = x_1.to(self.device)
                    moving = x_2.to(self.device)
                                        
                    # Forward pass through the registration model
                    t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                    
                    # Loss measures generator's ability to fool the discriminator
                    _, features_w1    = self.discriminator_net(w_1)
                    _, features_fixed = self.discriminator_net(TF.rotate(fixed, angle))
                    
                    # Compute the generator loss
                    generator_mse_penalty = gamma_value * self.disc_loss_fts(features_w1, features_fixed)
                    
                    # Computing the affine loss
                    affine_mse_loss    = self.mse_loss(w_0, fixed)

                    # Computing the elastic loss: Beta-VAE loss
                    reconstruction_loss = lambda_value * reconstruction_loss(fixed, w_1)
                    kl_divergence_loss  = kl_divergence(mu, log_var)
                    
                    # Total loss Beta-VAE valid (Generator)
                    loss_generator = affine_mse_loss + (reconstruction_loss + self.beta * kl_divergence_loss) + generator_mse_penalty
                    loss_pam_beta_vae_valid += loss_generator.item()
                    
                    # ----------- 1. Update the Discriminator -----------

                    # Measure discriminator's ability to classify real from generated samples
                    real, _ = self.discriminator_net(TF.rotate(fixed, angle))#(fixed)  # (w_0)
                    fake, _ = self.discriminator_net(w_1.detach())
                    b_size = real.shape
                    label_r  = torch.full(b_size, real_label, dtype=torch.float, device=self.device)
                    label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=self.device)

                    # Calculate loss
                    loss_d_real      = self.disc_loss_bce(real, label_r)
                    loss_d_fake      = self.disc_loss_bce(fake, label_f)
                    loss_discriminator = (loss_d_real + loss_d_fake) * 0.5
                    loss_disc_valid += loss_discriminator.item()

                    wandb.log({'epoch': i,
                            'Valid: Affine loss': affine_mse_loss.item(),
                            'Valid: Reconstruction loss': reconstruction_loss.item(),
                            'Valid: KL-divergence Loss': kl_divergence_loss.item(),
                            'Valid: Beta-VAE Loss': reconstruction_loss.item() + self.beta * kl_divergence_loss.item(),
                            'Valid: FTS Generator Loss': generator_mse_penalty.item(),
                            'Valid: Generator Total loss': loss_generator.item(),
                            'Valid: Discriminator loss': loss_discriminator,
                            })
                    
            
            # Save checkpoints
            if epoch % 10 == 0:
                name_pam = 'PAMModel_BetaVAE_Adversarial_' + str(epoch) + '.pth'
                name_dis = 'DisModel_BetaVAE_Adversarial_' + str(epoch) + '.pth'
                torch.save(self.net.state_dict(), os.path.join(self.checkpoints_folder, name_pam))
                torch.save(self.discriminator_net.state_dict(), os.path.join(self.checkpoints_folder, name_dis))
                print('Saving model')

            # Train loss per epoch
            data_loader_len         = len(self.train_dataloader)
            loss_disc_train        /= data_loader_len
            loss_pam_beta_vae_train/= data_loader_len

            # Valid loss per epoch
            data_loader_len         = len(self.valid_dataloader)
            loss_disc_valid        /= data_loader_len
            loss_pam_beta_vae_valid/= data_loader_len
        
            # Print the train and validation losses
            print("Train epoch : {}/{}, loss_PAM = {:.6f}, loss_Disc = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_train, loss_disc_train, self.beta)) # epoch + 1, n_epochs
            print("Valid epoch : {}/{}, loss_PAM = {:.6f}, loss_Disc = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_valid, loss_disc_valid ,self.beta))

            
    
    def train_WAE(self):

        # weights and biases
        wandb.init(project='WAE', entity='ljestaciocerquin')
        config = wandb.config
        wandb.watch(self.net)
        
        for epoch in range(self.start_epoch, self.n_epochs):
            # Affine losses for the training stage
            loss_affine_train      = 0
            loss_affine_sim_train  = 0
            loss_affine_reg_train  = 0
            
            # Elastic losses for the training stage
            loss_elastic_train     = 0
            loss_elastic_sim_train = 0
            loss_elastic_reg_train = 0
            
            # Beta-VAE loss for the training stage
            loss_wae_train         = 0
            loss_reconst_train     = 0
            loss_mmd_train         = 0
            
            # Total loss 
            loss_pam_wae_train     = 0
            
            # Affine losses for the validation stage
            loss_affine_valid      = 0
            loss_affine_sim_valid  = 0
            loss_affine_reg_valid  = 0
            
            # Elastic losses for the validation stage
            loss_elastic_valid     = 0
            loss_elastic_sim_valid = 0
            loss_elastic_reg_valid = 0
            
            # Beta-VAE loss for the validation stage
            loss_wae_valid        = 0
            loss_reconst_valid    = 0
            loss_mmd_valid        = 0
            
            # Total loss 
            loss_pam_wae_valid    = 0
            
            # Set the training mode
            self.net.train()
            
            for i, (x_1, x_2) in enumerate (self.train_dataloader):
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
                
                # zero-grad the net parameters
                self.optim.zero_grad()
                
                # Forward pass through the registration model
                t_0, w_0, t_1, w_1, z = self.net(fixed, moving)
                
                # Computing the affine loss
                sim_af, reg_af = total_loss(fixed, w_0, w_0)
                total_affine   = sim_af + reg_af
                
                loss_affine_sim_train += sim_af.item()
                loss_affine_reg_train += reg_af.item()
                loss_affine_train     += total_affine.item()
                
                               
                # Computing the elastic loss
                sim_ela, reg_ela = total_loss(fixed, w_1, t_1)
                total_elastic    = sim_ela + reg_ela
                
                loss_elastic_sim_train += sim_ela.item()
                loss_elastic_reg_train += reg_ela.item()
                loss_elastic_train     += total_elastic.item()
                
                # Computing the WAE loss
                z_fake = torch.autograd.Variable(torch.rand(fixed.size()[0], self.latent_dim) * 1)
                z_fake.to(self.device)
                recon_loss           = torch.nn.MSELoss(t_1, fixed)
                mmd_loss             = imq_kernel(z, z_fake, h_dim=self.latent_dim)
                total_loss_wae       = recon_loss + mmd_loss
                loss_reconst_train  += recon_loss.item()
                loss_mmd_train      += mmd_loss.item()
                loss_wae_train      += total_loss_wae.item()
                
                # Total loss
                loss = total_affine + total_elastic + total_loss_wae
                loss_pam_wae_train += loss
                
                # one backward pass
                loss.backward()
                
                # Update the parameters
                self.optim.step()
                
                
                # Display in tensorboard
                # ========
                wandb.log({'Iteration': epoch, 'Train: Similarity Affine loss': sim_af.item(),
                        'Train: Regression Affine loss': reg_af.item(),
                        'Train: Affine loss': total_affine.item(),
                        'Train: Similarity Elastic loss': sim_ela.item(),
                        'Train: Regression Elastic loss': reg_ela.item(),
                        'Train: Elastic loss':  total_elastic.item(),
                        'Train: Reconstruction loss': recon_loss.item(),
                        'Train: MMD Loss': mmd_loss.item(),
                        'Train: WAE Loss': total_loss_wae.item(),
                        'Train: Total loss': loss.item()})
            
            
            
            with torch.no_grad():
                self.net.eval()
                
                for i, (x_1, x_2) in enumerate (self.valid_dataloader):
                    fixed  = x_1.to(self.device)
                    moving = x_2.to(self.device)
                    
                    # Forward pass through the registration model
                    t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                    
                    # Computing the affine loss
                    #sim_af, reg_af = total_loss(fixed, w_0, w_0)
                    # total_affine   = sim_af + reg_af
                    
                    loss_affine_sim_valid += sim_af.item()
                    loss_affine_reg_valid += reg_af.item()
                    loss_affine_valid     += total_affine.item()
                    
                    # Computing the elastic loss
                    sim_ela, reg_ela = total_loss(fixed, w_1, t_1)
                    total_elastic    = sim_ela + reg_ela
                    
                    loss_elastic_sim_valid += sim_ela.item()
                    loss_elastic_reg_valid += reg_ela.item()
                    loss_elastic_valid     += total_elastic.item()
                    
                    # Computing the WAE loss
                    z_fake = torch.autograd.Variable(torch.rand(fixed.size()[0], self.latent_dim) * 1)
                    z_fake.to(self.device)
                    recon_loss           = torch.nn.MSELoss(t_1, fixed)
                    mmd_loss             = imq_kernel(z, z_fake, h_dim=self.latent_dim)
                    total_loss_wae       = recon_loss + mmd_loss
                    loss_reconst_valid  += recon_loss.item()
                    loss_mmd_valid      += mmd_loss.item()
                    loss_wae_valid      += total_loss_wae.item()
                    
                    # Total loss
                    loss = total_affine + total_elastic + total_loss_wae
                    loss_pam_wae_valid  += loss
                                     
                    
                    # Display in tensorboard
                    # ========
                    wandb.log({'Iteration': epoch, 'Valid: Similarity Affine loss': sim_af.item(),
                            'Valid: Regression Affine loss': reg_af.item(),
                            'Valid: Affine loss': total_affine.item(),
                            'Valid: Similarity Elastic loss': sim_ela.item(),
                            'Valid: Regression Elastic loss': reg_ela.item(),
                            'Valid: Elastic loss':  total_elastic.item(),
                            'Valid: Reconstruction loss': recon_loss.item(),
                            'Valid: MMD Loss': mmd_loss.item(),
                            'Valid: WAE Loss': total_loss_wae.item(),
                            'Valid: Total loss': loss.item()})
        
            
        # Compute the loss per epoch
        data_loader_len         = len(self.train_dataloader)
        loss_affine_sim_train  /= data_loader_len
        loss_affine_reg_train  /= data_loader_len
        loss_affine_train      /= data_loader_len
        loss_elastic_sim_train /= data_loader_len
        loss_elastic_reg_train /= data_loader_len
        loss_elastic_train     /= data_loader_len
        loss_reconst_train     /= data_loader_len
        loss_mmd_train         /= data_loader_len
        loss_wae_train         /= data_loader_len
        loss_pam_wae_train     /= data_loader_len
        
        # Save checkpoints
        if epoch % 10 == 0:
            name_pam = 'PAMModel_WAE_' + str(epoch) + '.pth'
            torch.save(self.net.state_dict(), os.path.join(self.checkpoints_folder, name_pam))
            print('Saving model')
        
         # Compute the loss per epoch
        data_loader_len         = len(self.valid_dataloader)
        loss_affine_sim_valid  /= data_loader_len
        loss_affine_reg_valid  /= data_loader_len
        loss_affine_valid      /= data_loader_len
        loss_elastic_sim_valid /= data_loader_len
        loss_elastic_reg_valid /= data_loader_len
        loss_elastic_valid     /= data_loader_len
        loss_reconst_valid     /= data_loader_len
        loss_mmd_valid         /= data_loader_len
        loss_wae_valid         /= data_loader_len
        loss_pam_wae_valid     /= data_loader_len
    
    
    def train_disentanglement_method(self):
        self.model_init()
        self.set_optimizer()
        self.load_dataloader()
        
        if self.add_discriminator:
            self.train_Beta_VAE_Adversarial()
        if self.model == 'WAE':
            self.train_WAE()
        elif self.model == 'Beta-VAE':
            self.train_Beta_VAE()
        else:
            NotImplementedError('only support WAE and Beta-VAE training!')
        