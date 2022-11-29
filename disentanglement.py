import os
import torch
import wandb
from   utils                        import cuda
from   utils                        import create_directory
from   utils                        import cuda_seeds
from   utils                        import weights_init
from   utils                        import read_train_data
from   losses                       import reconstruction_loss
from   losses                       import kl_divergence
from   losses                       import total_loss
from   losses                       import imq_kernel
from   networks.registration_model  import Registration_Beta_VAE
from   networks.registration_model  import Registration_Wasserstein_AE
from   torch.utils.data             import DataLoader
from   sklearn.model_selection      import train_test_split

class Disentanglement(object):
    def __init__(self, args):

        self.input_ch  = args.input_ch
        self.output_ch = args.output_ch
        self.data_dim  = args.data_dim
        self.z_dim     = args.z_dim
        self.img_shape = args.img_size
        self.filters   = [16, 32, 64, 128, 256]
        
        # Model
        self.model        = args.model
        self.decoder_dist = 'gaussian'
        
        # Model Parameters
        self.lr        = args.lr
        self.beta1     = args.beta1
        self.beta2     = args.beta2
        self.batch_size= args.batch_size
        
        # Path to save checkpoints
        self.checkpoints_folder = args.ckpt_dir
        
        # Device
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Beta value for Beta_VAE
        self.beta        = 4
        
        # Data folder
        self.data_folder = args.dset_dir
        
        self.n_epochs    = args.n_epochs
        self.start_epoch = 0
        
        # Directory to save checkpoints
        create_directory(self.checkpoints_folder)
        
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
        self.net = net(input_ch   = self.input_ch,
                            output_ch  = self.output_ch,
                            data_dim   = self.data_dim,
                            latent_dim = self.z_dim,
                            img_shape  = self.img_shape,
                            filters    = self.filters)
        
        # Init weights for the model
        self.net.apply(weights_init)
        
        # GPU computation
        self.net.to(self.device)
    
    
    def set_optimizer(self):
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr,
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
        train_losses = []
        valid_losses = []
        
        # weights and biases
        wandb.init(project='Beta-VAE', entity='ljestaciocerquin')
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
            loss_beta_vae_train    = 0
            loss_reconst_train     = 0
            loss_kl_diver_train    = 0
            
            # Total loss 
            loss_pam_beta_vae_train= 0
            
            # Affine losses for the validation stage
            loss_affine_valid      = 0
            loss_affine_sim_valid  = 0
            loss_affine_reg_valid  = 0
            
            # Elastic losses for the validation stage
            loss_elastic_valid     = 0
            loss_elastic_sim_valid = 0
            loss_elastic_reg_valid = 0
            
            # Beta-VAE loss for the validation stage
            loss_beta_vae_valid    = 0
            loss_reconst_valid     = 0
            loss_kl_diver_valid    = 0
            
            # Total loss 
            loss_pam_beta_vae_valid= 0
            
            # Set the training mode
            self.net.train()
            
            for i, (x_1, x_2) in enumerate (self.train_dataloader):
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
                
                # zero-grad the net parameters
                self.optim.zero_grad()
                
                # Forward pass through the registration model
                t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                
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
                
                # Computing the Beta-VAE loss
                recon_loss                        = reconstruction_loss(fixed, t_1, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, log_var)
                loss_beta_vae                     = recon_loss + self.beta*total_kld
                loss_reconst_train  += recon_loss.item()
                loss_kl_diver_train += total_kld.item()
                loss_beta_vae_train += loss_beta_vae.item()
                
                # Total loss
                loss = loss_affine_train + loss_elastic_train + loss_beta_vae_train
                loss_pam_beta_vae_train += loss
                
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
                        'Train: KL-divergence Loss': total_kld.item(),
                        'Train: KBeta-VAE Loss': total_kld.item(),
                        'Train: Total loss': loss.item()})
            
            
            
            with torch.no_grad():
                self.net.eval()
                
                for i, (x_1, x_2) in enumerate (self.valid_dataloader):
                    fixed  = x_1.to(self.device)
                    moving = x_2.to(self.device)
                    
                                        
                    # Forward pass through the registration model
                    t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                    
                    # Computing the affine loss
                    sim_af, reg_af = total_loss(fixed, w_0, w_0)
                    total_affine   = sim_af + reg_af
                    
                    loss_affine_sim_valid += sim_af.item()
                    loss_affine_reg_valid += reg_af.item()
                    loss_affine_valid     += total_affine.item()
                    
                                
                    # Computing the elastic loss
                    sim_ela, reg_ela = total_loss(fixed, w_1, t_1)
                    total_elastic    = sim_ela + reg_ela
                    
                    loss_elastic_sim_valid += sim_ela.item()
                    loss_elastic_reg_valid += reg_ela.item()
                    loss_elastic_valid     += total_elastic.item()
                    
                    # Computing the Beta-VAE loss
                    recon_loss                        = reconstruction_loss(fixed, t_1, self.decoder_dist)
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, log_var)
                    loss_beta_vae                     = recon_loss + self.beta*total_kld
                    loss_reconst_valid  += recon_loss.item()
                    loss_kl_diver_valid += total_kld.item()
                    loss_beta_vae_valid += loss_beta_vae.item()
                    
                    # Total loss
                    loss = loss_affine_valid + loss_elastic_valid + loss_beta_vae_valid
                    loss_pam_beta_vae_valid += loss
                    
                    # one backward pass
                    loss.backward()
                    
                    # Update the parameters
                    self.optim.step()
                    
                    
                    # Display in tensorboard
                    # ========
                    wandb.log({'Iteration': epoch, 'Valid: Similarity Affine loss': sim_af.item(),
                            'Valid: Regression Affine loss': reg_af.item(),
                            'Valid: Affine loss': total_affine.item(),
                            'Valid: Similarity Elastic loss': sim_ela.item(),
                            'Valid: Regression Elastic loss': reg_ela.item(),
                            'Valid: Elastic loss':  total_elastic.item(),
                            'Valid: Reconstruction loss': recon_loss.item(),
                            'Valid: KL-divergence Loss': total_kld.item(),
                            'Valid: KBeta-VAE Loss': total_kld.item(),
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
        loss_kl_diver_train    /= data_loader_len
        loss_beta_vae_train    /= data_loader_len
        loss_pam_beta_vae_train/= data_loader_len
        
        # Save checkpoints
        if epoch % 10 == 0:
            name_pam = 'PAMModel_BetaVAE_' + str(epoch) + '.pth'
            torch.save(self.net.state_dict(), os.path.join(self.checkpoints_folder, name_pam))
            print('Saving model')
    
    
    
    
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
            loss_wae_train    = 0
            loss_reconst_train     = 0
            loss_kl_diver_train    = 0
            
            # Total loss 
            loss_pam_beta_vae_train= 0
            
            # Affine losses for the validation stage
            loss_affine_valid      = 0
            loss_affine_sim_valid  = 0
            loss_affine_reg_valid  = 0
            
            # Elastic losses for the validation stage
            loss_elastic_valid     = 0
            loss_elastic_sim_valid = 0
            loss_elastic_reg_valid = 0
            
            # Beta-VAE loss for the validation stage
            loss_wae_valid    = 0
            loss_reconst_valid     = 0
            loss_kl_diver_valid    = 0
            
            # Total loss 
            loss_pam_wae_valid= 0
            
            # Set the training mode
            self.net.train()
            
            for i, (x_1, x_2) in enumerate (self.train_dataloader):
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
                
                # zero-grad the net parameters
                self.optim.zero_grad()
                
                # Forward pass through the registration model
                t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                
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
                
                # Computing the Beta-VAE loss
                recon_loss                        = reconstruction_loss(fixed, t_1, self.decoder_dist)
                mmd_loss                          = imq_kernel(fixed, t_1, h_dim=self.z_dim)
                loss_beta_vae                     = recon_loss + self.beta*total_kld
                loss_reconst_train  += recon_loss.item()
                loss_kl_diver_train += total_kld.item()
                loss_beta_vae_train += loss_beta_vae.item()
                
                # Total loss
                loss = loss_affine_train + loss_elastic_train + loss_beta_vae_train
                loss_pam_beta_vae_train += loss
                
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
                        'Train: KL-divergence Loss': total_kld.item(),
                        'Train: KBeta-VAE Loss': total_kld.item(),
                        'Train: Total loss': loss.item()})
            
            
            
            with torch.no_grad():
                self.net.eval()
                
                for i, (x_1, x_2) in enumerate (self.valid_dataloader):
                    fixed  = x_1.to(self.device)
                    moving = x_2.to(self.device)
                    
                                        
                    # Forward pass through the registration model
                    t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                    
                    # Computing the affine loss
                    sim_af, reg_af = total_loss(fixed, w_0, w_0)
                    total_affine   = sim_af + reg_af
                    
                    loss_affine_sim_valid += sim_af.item()
                    loss_affine_reg_valid += reg_af.item()
                    loss_affine_valid     += total_affine.item()
                    
                                
                    # Computing the elastic loss
                    sim_ela, reg_ela = total_loss(fixed, w_1, t_1)
                    total_elastic    = sim_ela + reg_ela
                    
                    loss_elastic_sim_valid += sim_ela.item()
                    loss_elastic_reg_valid += reg_ela.item()
                    loss_elastic_valid     += total_elastic.item()
                    
                    # Computing the Beta-VAE loss
                    recon_loss                        = reconstruction_loss(fixed, t_1, self.decoder_dist)
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, log_var)
                    loss_beta_vae                     = recon_loss + self.beta*total_kld
                    loss_reconst_valid  += recon_loss.item()
                    loss_kl_diver_valid += total_kld.item()
                    loss_beta_vae_valid += loss_beta_vae.item()
                    
                    # Total loss
                    loss = loss_affine_valid + loss_elastic_valid + loss_beta_vae_valid
                    loss_pam_beta_vae_valid += loss
                    
                    # one backward pass
                    loss.backward()
                    
                    # Update the parameters
                    self.optim.step()
                    
                    
                    # Display in tensorboard
                    # ========
                    wandb.log({'Iteration': epoch, 'Valid: Similarity Affine loss': sim_af.item(),
                            'Valid: Regression Affine loss': reg_af.item(),
                            'Valid: Affine loss': total_affine.item(),
                            'Valid: Similarity Elastic loss': sim_ela.item(),
                            'Valid: Regression Elastic loss': reg_ela.item(),
                            'Valid: Elastic loss':  total_elastic.item(),
                            'Valid: Reconstruction loss': recon_loss.item(),
                            'Valid: KL-divergence Loss': total_kld.item(),
                            'Valid: KBeta-VAE Loss': total_kld.item(),
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
        loss_kl_diver_train    /= data_loader_len
        loss_beta_vae_train    /= data_loader_len
        loss_pam_beta_vae_train/= data_loader_len
        
        # Save checkpoints
        if epoch % 10 == 0:
            name_pam = 'PAMModel_BetaVAE_' + str(epoch) + '.pth'
            torch.save(self.net.state_dict(), os.path.join(self.checkpoints_folder, name_pam))
            print('Saving model')