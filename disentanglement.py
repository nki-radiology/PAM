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
from   registration_dataset         import RegistrationDataSet
from   networks.registration_model  import Registration_Beta_VAE
from   networks.registration_model  import Registration_Wasserstein_AE
from   torch.utils.data             import DataLoader
from   sklearn.model_selection      import train_test_split
import matplotlib.pyplot            as     plt

class Disentanglement(object):
    def __init__(self, args):

        self.input_ch  = args.input_ch
        self.output_ch = args.output_ch
        self.data_dim  = args.data_dim
        self.z_dim     = args.z_dim
        self.group_num = args.group_num
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
        self.criterion = torch.nn.MSELoss()
        
        # Path to save checkpoints and results
        self.checkpoints_folder = args.ckpt_dir
        self.results_dir        = args.results_dir
        
        # Device
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Beta value for Beta_VAE
        self.beta        = 0
        
        # Data folder
        self.data_folder = args.dset_dir
        
        self.n_epochs    = args.n_epochs
        self.start_epoch = 1
        
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
        self.net = net(input_ch   = self.input_ch,
                            output_ch  = self.output_ch,
                            data_dim   = self.data_dim,
                            latent_dim = self.z_dim,
                            group_num  = self.group_num,
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
    
    
    def save_table(self, table_name, fixed_img, moving_img, affine_img, deformation_img, deformation_field):
        table = wandb.Table(columns=['Fixed Image', 'Moving Image', 'Affine Reg. Image', 'Deformation Reg. Image', 'Deformation Field'], allow_mixed_types = True)
    
        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.imshow(fixed_img[0].squeeze().detach().cpu().numpy())
        plt.savefig(self.results_dir + "fixed_image.jpg")
        plt.close()
        
        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.imshow(moving_img[0].squeeze().detach().cpu().numpy())
        plt.savefig(self.results_dir + "moving_image.jpg")
        plt.close()

        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.imshow(affine_img[0].squeeze().detach().cpu().numpy())
        plt.savefig(self.results_dir + "affine_reg_image.jpg")
        plt.close()

        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.imshow(deformation_img[0].squeeze().detach().cpu().numpy())
        plt.savefig(self.results_dir + "deformation_reg_image.jpg")
        plt.close()
        
        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.imshow(deformation_field[0].squeeze().detach().cpu().numpy())
        plt.savefig(self.results_dir + "deformation_field.jpg")
        plt.close()
        
        table.add_data(
            wandb.Image(plt.imread(self.results_dir + "fixed_image.jpg")),
            wandb.Image(plt.imread(self.results_dir + "moving_image.jpg")),
            wandb.Image(plt.imread(self.results_dir + "affine_reg_image.jpg")),
            wandb.Image(plt.imread(self.results_dir + "deformation_reg_image.jpg")),
            wandb.Image(plt.imread(self.results_dir + "deformation_field.jpg"))
        )
        
        wandb.log({table_name: table})

        
    def train_Beta_VAE(self):
        
        # weights and biases
        wandb.init(project='Beta-VAE', entity='ljestaciocerquin')
        config = wandb.config
        wandb.watch(self.net, log="all")
        
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
            lambda_value           = 0.001
            
            # Set the training mode
            self.net.train()
            
            
            if epoch == 50:
                self.beta = 1e-10
            
            elif epoch > 50:
                if self.beta < 4:
                    self.beta = self.beta * 2
                else:
                    self.beta = 4        
                                        
            
            for i, (x_1, x_2) in enumerate (self.train_dataloader):
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
               
                
                # zero-grad the net parameters
                self.optim.zero_grad()
                
                # Forward pass through the registration model
                t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                
                
                # Computing the affine loss
                #sim_af, reg_af = total_loss(fixed, w_0, w_0)
                #total_affine  = sim_af #+ reg_af
                #total_affine   = reg_af
                total_affine  = self.criterion(w_0, fixed)
                
                #loss_affine_sim_train += sim_af.item() # This one : Affine!
                #loss_affine_reg_train += reg_af.item()
                loss_affine_train     += total_affine.item()
                
                               
                # Computing the elastic loss
                #sim_ela, reg_ela = total_loss(fixed, w_1, t_1)
                #total_elastic    = sim_ela #+ reg_ela
                
                #loss_elastic_sim_train += sim_ela.item()
                #loss_elastic_reg_train += reg_ela.item()
                #loss_elastic_train     += total_elastic.item()
                
                # Computing the Beta-VAE loss
                recon_loss                        = lambda_value * reconstruction_loss(fixed, t_1, self.decoder_dist) #0.00001 * reconstruction_loss(fixed, t_1, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, log_var)
                total_loss_beta_vae               = recon_loss + self.beta*total_kld
                loss_reconst_train  += recon_loss.item()
                loss_kl_diver_train += total_kld.item()
                loss_beta_vae_train += total_loss_beta_vae.item()
                
                # Total loss
                loss = total_affine + total_loss_beta_vae #total_affine + total_elastic + total_loss_beta_vae
                loss_pam_beta_vae_train += loss.item()
                
                # one backward pass
                loss.backward()
                
                # Update the parameters
                self.optim.step()
                
                
                # Display in tensorboard
                # ========
                """wandb.log({'Iteration': epoch, 'Train: Similarity Affine loss': sim_af.item(),
                        'Train: Regression Affine loss': reg_af.item(),
                        'Train: Affine loss': total_affine.item(),
                        'Train: Similarity Elastic loss': sim_ela.item(),
                        'Train: Regression Elastic loss': reg_ela.item(),
                        'Train: Elastic loss':  total_elastic.item(),
                        'Train: Reconstruction loss': recon_loss.item(),
                        'Train: KL-divergence Loss': total_kld.item(),
                        'Train: Beta-VAE Loss': total_loss_beta_vae.item(),
                        'Train: Total loss': loss.item()})
            
                """
                
            
            with torch.no_grad():
                self.net.eval()
                
                for i, (x_1, x_2) in enumerate (self.valid_dataloader):
                    fixed  = x_1.to(self.device)
                    moving = x_2.to(self.device)
                  
                    
                                        
                    # Forward pass through the registration model
                    t_0, w_0, t_1, w_1, mu, log_var = self.net(fixed, moving)
                    
                    # Computing the affine loss
                    #sim_af, reg_af = total_loss(fixed, w_0, w_0)
                    total_affine  = self.criterion(w_0, fixed) #sim_af # + reg_af
                    #total_affine   = reg_af
                    
                    #loss_affine_sim_valid += sim_af.item()
                    #loss_affine_reg_valid += reg_af.item()
                    loss_affine_valid     += total_affine.item()
                    
                                
                    # Computing the elastic loss
                    #sim_ela, reg_ela = total_loss(fixed, w_1, t_1)
                    #total_elastic    = sim_ela + reg_ela
                    
                    #loss_elastic_sim_valid += sim_ela.item()
                    #loss_elastic_reg_valid += reg_ela.item()
                    #loss_elastic_valid     += total_elastic.item()
                    
                    # Computing the Beta-VAE loss
                    recon_loss                        = lambda_value * reconstruction_loss(fixed, t_1, self.decoder_dist) #0.00001 * reconstruction_loss(fixed, t_1, self.decoder_dist)
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, log_var)
                    total_loss_beta_vae               = recon_loss + self.beta*total_kld
                    loss_reconst_valid  += recon_loss.item()
                    loss_kl_diver_valid += total_kld.item()
                    loss_beta_vae_valid += total_loss_beta_vae.item()
                    
                    # Total loss
                    loss                     = total_affine + total_loss_beta_vae #total_affine + total_elastic + total_loss_beta_vae
                    loss_pam_beta_vae_valid += loss.item()
                    

                    # Display in tensorboard
                    # ========
                    """wandb.log({'Iteration': epoch, 'Valid: Similarity Affine loss': sim_af.item(),
                            'Valid: Regression Affine loss': reg_af.item(),
                            'Valid: Affine loss': total_affine.item(),
                            'Valid: Similarity Elastic loss': sim_ela.item(),
                            'Valid: Regression Elastic loss': reg_ela.item(),
                            'Valid: Elastic loss':  total_elastic.item(),
                            'Valid: Reconstruction loss': recon_loss.item(),
                            'Valid: KL-divergence Loss': total_kld.item(),
                            'Valid: Beta-VAE Loss': total_loss_beta_vae.item(),
                            'Valid: Total loss': loss.item()})"""
                    
                   
            
            # Compute the loss per epoch
            data_loader_len         = len(self.train_dataloader)
            #loss_affine_sim_train  /= data_loader_len
            #loss_affine_reg_train  /= data_loader_len
            loss_affine_train      /= data_loader_len
            #loss_elastic_sim_train /= data_loader_len
            #loss_elastic_reg_train /= data_loader_len
            #loss_elastic_train     /= data_loader_len
            loss_reconst_train     /= data_loader_len
            loss_kl_diver_train    /= data_loader_len
            loss_beta_vae_train    /= data_loader_len
            loss_pam_beta_vae_train/= data_loader_len
        
            # Save checkpoints
            if epoch % 10 == 0:
                name_pam = 'PAMModel_BetaVAE_' + str(epoch) + '.pth'
                torch.save(self.net.state_dict(), os.path.join(self.checkpoints_folder, name_pam))
                print('Saving model')

            # Compute the loss per epoch
            data_loader_len         = len(self.valid_dataloader)
            #loss_affine_sim_valid  /= data_loader_len
            #loss_affine_reg_valid  /= data_loader_len
            loss_affine_valid      /= data_loader_len
            #loss_elastic_sim_valid /= data_loader_len
            #loss_elastic_reg_valid /= data_loader_len
            #loss_elastic_valid     /= data_loader_len
            loss_reconst_valid     /= data_loader_len
            loss_kl_diver_valid    /= data_loader_len
            loss_beta_vae_valid    /= data_loader_len
            loss_pam_beta_vae_valid/= data_loader_len
        
        
            # Display in tensorboard
            # ========
            wandb.log({'epoch': epoch+1, #'Train: Similarity Affine loss': loss_affine_sim_train,
                            #'Train: Regression Affine loss': loss_affine_reg_train,
                            'Train: Affine loss': loss_affine_train,
                            #'Train: Similarity Elastic loss': loss_elastic_sim_train,
                            #'Train: Regression Elastic loss': loss_elastic_reg_train,
                            #'Train: Elastic loss':  loss_elastic_train,
                            'Train: Reconstruction loss': loss_reconst_train,
                            'Train: KL-divergence Loss': loss_kl_diver_train,
                            'Train: Beta-VAE Loss': loss_beta_vae_train,
                            'Train: Total loss': loss_pam_beta_vae_train,
                            #'Valid: Similarity Affine loss': loss_affine_sim_valid,
                            #'Valid: Regression Affine loss': loss_affine_reg_valid,
                            'Valid: Affine loss': loss_affine_valid,
                            #'Valid: Similarity Elastic loss': loss_elastic_sim_valid,
                            #'Valid: Regression Elastic loss': loss_elastic_reg_valid,
                            #'Valid: Elastic loss':  loss_elastic_valid,
                            'Valid: Reconstruction loss': loss_reconst_valid,
                            'Valid: KL-divergence Loss': loss_kl_diver_valid,
                            'Valid: Beta-VAE Loss': loss_beta_vae_valid,
                            'Valid: Total loss': loss_pam_beta_vae_valid})
        
            # Print the train and validation losses
            print("Train epoch : {}/{}, loss_PAM = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_train, self.beta)) # epoch + 1, n_epochs
            print("Valid epoch : {}/{}, loss_PAM = {:.6f}, beta_value = {:.6f}".format(epoch, self.n_epochs, loss_pam_beta_vae_valid, self.beta))
            
            
    
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
                z_fake = torch.autograd.Variable(torch.rand(fixed.size()[0], self.z_dim) * 1)
                z_fake.to(self.device)
                recon_loss           = torch.nn.MSELoss(t_1, fixed)
                mmd_loss             = imq_kernel(z, z_fake, h_dim=self.z_dim)
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
                    z_fake = torch.autograd.Variable(torch.rand(fixed.size()[0], self.z_dim) * 1)
                    z_fake.to(self.device)
                    recon_loss           = torch.nn.MSELoss(t_1, fixed)
                    mmd_loss             = imq_kernel(z, z_fake, h_dim=self.z_dim)
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
        
        if self.model == 'WAE':
            self.train_WAE()
        elif self.model == 'Beta-VAE':
            self.train_Beta_VAE()
        else:
            NotImplementedError('only support WAE and Beta-VAE training!')
        