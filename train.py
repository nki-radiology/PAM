import os
import torch
import wandb
from   utils                        import create_directory
from   utils                        import cuda_seeds
from   utils                        import weights_init
from   utils                        import frozen_params
from   utils                        import free_params
from   utils                        import read_2D_train_data
from   utils                        import read_3D_train_data
from   utils                        import save_images_weights_and_biases
from   losses                       import *
from   registration_dataset         import Registration2DDataSet
from   registration_dataset         import Registration3DDataSet
from   networks.registration_model  import Registration_Beta_VAE
from   networks.registration_model  import Registration_Wasserstein_AE
from   networks.discriminator       import Discriminator
from   networks.discriminator       import Discriminator_Linear
from   torch.utils.data             import DataLoader
from   sklearn.model_selection      import train_test_split
import numpy                        as     np
import torch.nn.functional          as F   
from torch.optim.lr_scheduler import StepLR          
from random import randint

class Train(object):
    def __init__(self, args):

        self.input_ch   = args.input_ch
        self.input_dim  = args.input_dim
        self.latent_dim = args.latent_dim
        self.output_ch  = args.output_ch
        self.group_num  = args.group_num
        self.filters    = args.filters
        self.filters_disc = args.filters_disc
        self.input_ch_discriminator = args.input_ch_d
        
        # Model
        self.model             = args.model
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
        
        # Accumulation itertator for gradient accumulation
        self. accum_iter_batch = args.accum_iter_b
        
        # Values of the regularization parameters
        self.alpha_value  = args.alpha_value            # regularization for the penalty loss
        self.beta_value   = args.beta_value             # regularization for the KL-divergence loss
        self.beta_value_max = args.beta_value_max       # regularization for the KL-divergence loss
        self.gamma_value  = args.gamma_value            # regularization for the discriminator (feature matching loss: MSE)
        self.lambda_value = args.lambda_value           # regularization for the reconstruction loss
        self.sigma_value  = args.sigma_value            # regularizer for WAE
    
        # Data folder
        self.data_folder = args.dset_dir
        
        # Number of epochs to train the model
        self.n_epochs    = args.n_epochs
        self.start_epoch = args.start_ep #1
        
        # Variables to save weights and biases images
        self.fixed_draw  = None
        self.moving_draw = None
        self.w_0_draw    = None
        self.w_1_draw    = None
        
        # Weights & biases projects name and entity
        self.proj_wae              = args.proj_wae
        self.proj_bvae_vanilla     = args.proj_bvae_vanilla
        self.proj_bvae_adversarial = args.proj_bvae_adversarial
        self.entity_wb             = args.entity_wb
        
        # Directory to save checkpoints
        create_directory(self.checkpoints_folder)
        create_directory(self.results_dir)
        
        # Cuda Seeds for reproducibility
        cuda_seeds()
        
    
    def model_init(self):
        if self.model == 'WAE':
            net = Registration_Wasserstein_AE
        elif self.model == 'Beta-VAE':
            net = Registration_Beta_VAE
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
                input_dim  = self.input_dim,
                group_num  = self.group_num,
                filters    = self.filters_disc)  
            
            self.discriminator_net.to(self.device)
            # Handle multi-gpu if desired
            if (self.device.type == 'cuda') and (self.num_gpus > 1):
                self.discriminator_net = torch.nn.DataParallel(self.discriminator_net, list(range(self.num_gpus)))
            self.discriminator_net.apply(weights_init)
            self.disc_loss_bce = torch.nn.BCELoss()
            self.disc_loss_fts = torch.nn.MSELoss()
        
        if self.model == 'WAE':
            self.discriminator_net = Discriminator_Linear(latent_dim=self.latent_dim)
            self.discriminator_net.to(self.device)
            # Handle multi-gpu if desired
            if (self.device.type == 'cuda') and (self.num_gpus > 1):
                self.discriminator_net = torch.nn.DataParallel(self.discriminator_net, list(range(self.num_gpus)))
            self.discriminator_net.apply(weights_init)
            self.disc_loss_bce = torch.nn.BCEWithLogitsLoss()
            self.disc_loss_rec = torch.nn.MSELoss()

        
    def set_optimizer(self):
        # self.optim = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        if self.add_discriminator or self.model == 'WAE':
            self.optim_disc = torch.optim.Adam(self.discriminator_net.parameters(), lr=self.lr, 
                                            betas=(self.beta1, self.beta2))
        
                    
    def load_dataloader(self):
        # Dataset Path 
        if len(self.input_dim) == 2:
            filenames = read_2D_train_data(self.data_folder)
        else:
            filenames = read_3D_train_data(self.data_folder)
       
        # Random seed
        random_seed = 42

        # Split dataset into training set and validation set
        train_size  = 0.8
        inputs_train, inputs_valid = train_test_split(
            filenames, random_state=random_seed, train_size=train_size, shuffle=True
        )

        print("total: ", len(filenames), " train: ", len(inputs_train), " valid: ", len(inputs_valid))
        
        if len(self.input_dim) == 2:
            registration_dataset = Registration2DDataSet
        else:
            registration_dataset = Registration3DDataSet

        # Training and validation dataset
        train_dataset = registration_dataset(path_dataset = inputs_train,
                                             input_dim    = self.input_dim,
                                             transform    = None)
        valid_dataset = registration_dataset(path_dataset = inputs_valid,
                                             input_dim    = self.input_dim,
                                             transform    = None)

        # Training and validation dataloader
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=True)
    
    
           
    def train_Beta_VAE(self):
        
        # weights and biases
        wandb.init(project=self.proj_bvae_vanilla, entity=self.entity_wb)
        config = wandb.config
        wandb.watch(self.net, log="all")
        
        for epoch in range(self.start_epoch, self.n_epochs):
                        
            # Total loss 
            loss_pam_beta_vae_train= 0
            
            # Total loss 
            loss_pam_beta_vae_valid= 0
            
            # Set the training mode
            self.net.train()
            
            if epoch >= 50: 
                self.beta_value = np.maximum(self.beta_value, 1e-10)
                self.beta_value = np.minimum(self.beta_value*2, self.beta_value_max)      
            
            for batch_idx, (x_1, x_2) in enumerate (self.train_dataloader):
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
                
                # zero-grad the net parameters
                self.optim.zero_grad()
               
                # Forward pass through the registration model
                t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                
                # Computing the affine loss
                affine_mse_loss     = self.mse_loss(w_0, fixed)
                penalty_affine_loss = self.energy_loss.energy_loss(t_0)
                
                # Computing the elastic loss: Beta-VAE loss
                elastic_mse_loss     = self.mse_loss(w_1, fixed)  
                penalty_elastic_loss = self.energy_loss.energy_loss(t_1)
                kl_divergence_loss   = kl_divergence(mu, log_var)
                              
                
                loss = self.alpha_value  * (penalty_affine_loss + penalty_elastic_loss) + \
                       self.beta_value   * (kl_divergence_loss) +\
                       self.lambda_value * (elastic_mse_loss + affine_mse_loss)
                loss_pam_beta_vae_train += loss.item()
                
                print(' --------------------  -------------------- KL Loss: ', kl_divergence_loss.item())
                
                # one backward pass
                loss.backward()
                
                # Update the parameters
                self.optim.step()
                
                # Gradient accumulation
                #if ( (batch_idx + 1) % self.accum_iter_batch == 0) or (batch_idx + 1 == len(self.train_dataloader)):
                # self.optim.zero_grad()
                # self.optim.step()
                # loss = loss / self.accum_iter_batch

                # Weights and biases visualization
                wandb.log({'Iteration': batch_idx,
                        'Train: Affine MSE loss': affine_mse_loss.item(),
                        'Train: Penalty Affine loss': penalty_affine_loss.item(),
                        'Train: Elastic MSE loss': elastic_mse_loss.item(),
                        'Train: Penalty Elastic loss': penalty_elastic_loss.item(),
                        'Train: KL-divergence Loss': kl_divergence_loss.item(),
                        'Train: Total loss': loss.item()})
                
            
            with torch.no_grad():
                self.net.eval()
                
                for batch_idx, (x_1, x_2) in enumerate (self.valid_dataloader):
                    fixed  = x_1.to(self.device)
                    moving = x_2.to(self.device)
                                        
                    # Forward pass through the registration model
                    t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                    
                    # Computing the affine loss
                    affine_mse_loss     = self.mse_loss(w_0, fixed)
                    penalty_affine_loss = self.energy_loss.energy_loss(t_0)
                    
                    # Computing the elastic loss: Beta-VAE loss
                    elastic_mse_loss     = self.mse_loss(w_1, fixed)  
                    penalty_elastic_loss = self.energy_loss.energy_loss(t_1)
                    kl_divergence_loss   = kl_divergence(mu, log_var)
                    
                    # Total loss
                    loss = self.alpha_value  * (penalty_affine_loss + penalty_elastic_loss) + \
                           self.beta_value   * (kl_divergence_loss) +\
                           self.lambda_value * (elastic_mse_loss + affine_mse_loss)
                    loss_pam_beta_vae_valid += loss.item()
                    
                    # Weights and biases visualization
                    wandb.log({ 'Iteration': batch_idx,
                                'Valid: Affine MSE loss': affine_mse_loss.item(),
                                'Valid: Penalty Affine loss': penalty_affine_loss.item(),
                                'Valid: Elastic MSE loss': elastic_mse_loss.item(),
                                'Valid: Penalty Elastic loss': penalty_elastic_loss.item(),
                                'Valid: KL-divergence Loss': kl_divergence_loss.item(),
                                'Valid: Total loss': loss.item()})
                    
                    self.fixed_draw  = fixed[0]
                    self.moving_draw = moving[0]
                    self.w_0_draw    = w_0[0]
                    self.w_1_draw    = w_1[0]

            # Visualization of images
            save_images_weights_and_biases('Validation Images', self.results_dir, self.fixed_draw, self.moving_draw, self.w_0_draw, self.w_1_draw)
        
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
            print("Train epoch : {}/{}, loss_PAM = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_train, self.beta_value)) 
            print("Valid epoch : {}/{}, loss_PAM = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_valid, self.beta_value))



    def train_Beta_VAE_Adversarial(self):
        
        # weights and biases
        wandb.init(project=self.proj_bvae_adversarial, entity=self.entity_wb)
        config = wandb.config
        wandb.watch(self.net, log="all")
                      
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
                self.beta_value = np.maximum(self.beta_value, 1e-10)
                self.beta_value = np.minimum(self.beta_value*2, self.beta_value_max)  
            
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
                _, features_fixed = self.discriminator_net(fixed)#(TF.rotate(fixed, angle)) 
                
                # Compute generator loss
                generator_mse_penalty = self.disc_loss_fts(features_w1, features_fixed)
                
                # Computing the affine loss
                affine_mse_loss     = self.nn_loss.pearson_correlation(fixed, w_0) #self.mse_loss(w_0, fixed)
                penalty_affine_loss = self.energy_loss.energy_loss(t_0)
                
                # Computing the elastic loss: Beta-VAE loss
                elastic_mse_loss     = self.nn_loss.pearson_correlation(fixed, w_1) #self.mse_loss(w_1, fixed) 
                penalty_elastic_loss = self.energy_loss.energy_loss(t_1)
                kl_divergence_loss   = kl_divergence(mu, log_var)

                # Total loss Beta-VAE train
                loss_generator =    self.alpha_value  * (penalty_affine_loss + penalty_elastic_loss) + \
                                    self.beta_value   * (kl_divergence_loss) +\
                                    self.lambda_value * (elastic_mse_loss + affine_mse_loss) + \
                                    self.gamma_value  * generator_mse_penalty
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
                real, _  = self.discriminator_net(fixed)#(TF.rotate(fixed, angle))  
                fake, _  = self.discriminator_net(w_1.detach())
                b_size   = real.shape
                label_r  = torch.full(b_size, real_label, dtype=torch.float, device=self.device)
                label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=self.device)
                
                # Compute discriminator loss
                loss_d_real = self.disc_loss_bce(real, label_r)
                loss_d_fake = self.disc_loss_bce(fake, label_f)
                loss_discriminator = (loss_d_real + loss_d_fake) * 0.5
                loss_disc_train   += loss_discriminator.item()
                
                # one backward pass
                loss_discriminator.backward()

                # Update Discriminator
                self.optim_disc.step()
                
                # Reinit the affine network weights
                if loss_discriminator.item() < 1e-5:  # 
                    self.discriminator_net.apply(weights_init)
                    print("Reloading discriminator weights")
                
                wandb.log({'Iteration': i,
                            'Train: Affine MSE loss': affine_mse_loss.item(),
                            'Train: Penalty Affine loss': penalty_affine_loss.item(),
                            'Train: Elastic MSE loss': elastic_mse_loss.item(),
                            'Train: Penalty Elastic loss': penalty_elastic_loss.item(),
                            'Train: KL-divergence Loss': kl_divergence_loss.item(),
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
                    _, features_fixed = self.discriminator_net(fixed)#(TF.rotate(fixed, angle))
                    
                    # Compute the generator loss
                    generator_mse_penalty = self.disc_loss_fts(features_w1, features_fixed)
                    
                    # Computing the affine loss
                    affine_mse_loss     = self.nn_loss.pearson_correlation(fixed, w_0) #self.mse_loss(w_0, fixed)
                    penalty_affine_loss = self.energy_loss.energy_loss(t_0)

                    # Computing the elastic loss: Beta-VAE loss
                    elastic_mse_loss     = self.nn_loss.pearson_correlation(fixed, w_1) #self.mse_loss(w_1, fixed)
                    penalty_elastic_loss = self.energy_loss.energy_loss(t_1)
                    kl_divergence_loss   = kl_divergence(mu, log_var)
                    
                    # Total loss Beta-VAE valid (Generator)                   
                    loss_generator = self.alpha_value  * (penalty_affine_loss + penalty_elastic_loss) + \
                                     self.beta_value   * (kl_divergence_loss) +\
                                     self.lambda_value * (elastic_mse_loss + affine_mse_loss) + \
                                     self.gamma_value  * generator_mse_penalty
                    loss_pam_beta_vae_valid += loss_generator.item()
                    
                    
                    # ----------- 1. Update the Discriminator -----------

                    # Measure discriminator's ability to classify real from generated samples
                    real, _ = self.discriminator_net(fixed)#(TF.rotate(fixed, angle))
                    fake, _ = self.discriminator_net(w_1.detach())
                    b_size = real.shape
                    label_r  = torch.full(b_size, real_label, dtype=torch.float, device=self.device)
                    label_f  = torch.full(b_size, fake_label, dtype=torch.float, device=self.device)

                    # Calculate loss
                    loss_d_real      = self.disc_loss_bce(real, label_r)
                    loss_d_fake      = self.disc_loss_bce(fake, label_f)
                    loss_discriminator = (loss_d_real + loss_d_fake) * 0.5
                    loss_disc_valid += loss_discriminator.item()

                    wandb.log({ 'Iteration': i,
                                'Valid: Affine MSE loss': affine_mse_loss.item(),
                                'Valid: Penalty Affine loss': penalty_affine_loss.item(),
                                'Valid: Elastic MSE loss': elastic_mse_loss.item(),
                                'Valid: Penalty Elastic loss': penalty_elastic_loss.item(),
                                'Valid: KL-divergence Loss': kl_divergence_loss.item(),
                                'Valid: FTS Generator Loss': generator_mse_penalty.item(),
                                'Valid: Generator Total loss': loss_generator.item(),
                                'Valid: Discriminator loss': loss_discriminator,
                        })
                    
                    self.fixed_draw = fixed[0]
                    self.moving_draw = moving[0]
                    self.w_0_draw    = w_0[0]
                    self.w_1_draw    = w_1[0]

            # Visualization of images
            save_images_weights_and_biases('Validation Images', self.results_dir, self.fixed_draw, self.moving_draw, self.w_0_draw, self.w_1_draw)
        
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
            print("Train epoch : {}/{}, loss_PAM = {:.6f}, loss_Disc = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_train, loss_disc_train, self.beta_value)) # epoch + 1, n_epochs
            print("Valid epoch : {}/{}, loss_PAM = {:.6f}, loss_Disc = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_valid, loss_disc_valid, self.beta_value))

            
    
    def train_WAE(self):

        # weights and biases
        wandb.init(project=self.proj_wae, entity=self.entity_wb)
        config = wandb.config
        wandb.watch(self.net)
        
        # Establish convention for real and fake labels during training
        one = torch.tensor(1, dtype=torch.float, device=self.device)
        eps = 1e-5
        
        for epoch in range(self.start_epoch, self.n_epochs):

            # Train Total loss 
            loss_pam_wae_train = 0
            loss_disc_train    = 0
            
            # ValidTotal loss 
            loss_pam_wae_valid = 0
            loss_disc_valid    = 0
            
            # Set the training mode
            self.net.train()
            self.discriminator_net.train()
                        
            for i, (x_1, x_2) in enumerate (self.train_dataloader):
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
                
                # zero-grad the net parameters
                self.optim.zero_grad()
                self.optim_disc.zero_grad()
                
                # Train Discriminator
                frozen_params(self.net)
                free_params(self.discriminator_net)
                
                z_fake = torch.rand(fixed.size()[0], self.latent_dim, device=self.device) * self.sigma_value
                d_fake = self.discriminator_net(z_fake)
                
                # Forward pass through the registration model
                t_0, w_0, t_1, w_1, z_real = self.net(fixed, moving)
                d_real                     = self.discriminator_net(z_real)
                loss_d_fake = self.lambda_value * torch.log(d_fake + eps).mean()
                loss_d_real = self.lambda_value * torch.log(1 - d_real + eps).mean()
                loss_d_fake.backward(-one)
                loss_d_real.backward(-one)
                print('loss_d_fake: ', loss_d_fake)
                print('loss_d_real: ', loss_d_real)
                
                loss_disc_train += (loss_d_fake.item() + loss_d_real.item())
                self.optim_disc.step()
                
                # Train Generator
                free_params(self.net)
                frozen_params(self.discriminator_net)
                
                _, _, _, w_1, z_real  = self.net(fixed, moving)
                d_real                = self.discriminator_net(z_real)
                reconstruction_loss  = self.disc_loss_rec(w_1, fixed).mean()
                print(z_real)
                print('z_real: ', z_real.min(), z_real.max(), (torch.log(z_real + eps)).mean())
                d_loss               = self.lambda_value * (torch.log(z_real + eps)).mean()
                reconstruction_loss.backward(one, retain_graph=True)
                d_loss.backward(-one)
                print('reconstruction_loss: ', reconstruction_loss)
                print('d_loss: ', d_loss)
                
                # one backward pass
                # loss_generator.backward()
                loss_pam_wae_train  += reconstruction_loss.item() + d_loss.item()
                
                # Update the parameters
                self.optim.step()
                
                # Weights and biases visualization
                # ========
                wandb.log({'Iteration': i, 
                        'Train: Reconstruction loss': reconstruction_loss.item(),
                        'Train: MMD Loss': d_loss.item(),
                        'Train: Discriminator Loss': loss_d_fake.item() + loss_d_real.item()})
            
            
            with torch.no_grad():
                self.net.eval()
                self.discriminator_net.eval()
                
                for i, (x_1, x_2) in enumerate (self.valid_dataloader):
                    fixed  = x_1.to(self.device)
                    moving = x_2.to(self.device)
                    
                     # Train Discriminator
                    z_fake = torch.rand(fixed.size()[0], self.latent_dim, device=self.device) * self.sigma_value
                    d_fake = self.discriminator_net(z_fake)
                    
                    # Forward pass through the registration model
                    t_0, w_0, t_1, w_1, z_real = self.net(fixed, moving)
                    d_real                     = self.discriminator_net(z_real)
                    loss_d_fake = self.lambda_value * torch.log(d_fake + eps).mean()
                    loss_d_real = self.lambda_value * torch.log(1 - d_real + eps).mean()
                    loss_disc_valid += (loss_d_fake.item() + loss_d_real.item())
                
                    # Train Generator
                    t_0, w_0, t_1, w_1, z_real  = self.net(fixed, moving)
                    d_real                      = self.discriminator_net(z_real)
                    reconstruction_loss  = self.disc_loss_rec(w_1, fixed).mean()
                    d_loss               = self.lambda_value * (torch.log(z_real + eps)).mean()
                    loss_pam_wae_valid  += reconstruction_loss.item() + d_loss.item()
                
                    
                    # Weights and biases visualization
                    # ========
                    wandb.log({'Iteration': i, 
                            'Valid: Reconstruction loss': reconstruction_loss.item(),
                            'Valid: MMD Loss': d_loss.item(),
                            'Valid: Discriminator Loss': loss_d_fake.item() + loss_d_real.item()})
                        
                    self.fixed_draw = fixed[0]
                    self.moving_draw = moving[0]
                    self.w_0_draw    = w_0[0]
                    self.w_1_draw    = w_1[0]

            # Visualization of images
            save_images_weights_and_biases('Validation Images', self.results_dir, self.fixed_draw, self.moving_draw, self.w_0_draw, self.w_1_draw)
            
            
            # Save checkpoints
            if epoch % 10 == 0:
                name_pam = 'PAMModel_WAE_' + str(epoch) + '.pth'
                torch.save(self.net.state_dict(), os.path.join(self.checkpoints_folder, name_pam))
                print('Saving model')
                
            # Train loss per epoch
            loss_pam_wae_train = loss_pam_wae_train / len(self.train_dataloader)
            loss_disc_train    = loss_disc_train / len(self.train_dataloader)
            
            # Valid loss per epoch
            loss_pam_wae_valid = loss_pam_wae_valid / len(self.valid_dataloader)
            loss_disc_valid    = loss_disc_valid / len(self.train_dataloader)

            # Print the train and validation losses
            print("Train epoch : {}/{}, loss_PAM_WAE = {:.6f}, loss_Disc_WAE = {:.6f}".format(epoch, self.n_epochs, loss_pam_wae_train, loss_disc_train)) 
            print("Valid epoch : {}/{}, loss_PAM_WAE = {:.6f}, loss_Disc_WAE = {:.6f}".format(epoch, self.n_epochs, loss_pam_wae_valid, loss_disc_valid))
    
    

    def train_disentanglement_method(self):
        self.model_init()
        self.set_optimizer()
        self.load_dataloader()

        if self.add_discriminator:
            self.train_Beta_VAE_Adversarial()
            print('-------------- Running Adversarial Beta-VAE Model --------------')
        if self.model == 'WAE':
            self.train_WAE()
            print('----------------------- Running WAE Model ----------------------')
        elif self.model == 'Beta-VAE':
            self.train_Beta_VAE()
            print('-------------------- Running Beta-VAE Model --------------------')
        else:
            NotImplementedError('only support WAE and Beta-VAE training!')
        