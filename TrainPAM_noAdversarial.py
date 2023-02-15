# Importing general lybraries
import os
import wandb                           
import pandas                     as     pd
import torch.nn                   as     nn
import torchvision.transforms     as     T
from   PIL                        import Image
from   tqdm                       import tqdm
from   pathlib                    import Path
from   torch.utils.data           import DataLoader
from   sklearn.model_selection    import train_test_split
import matplotlib.pyplot          as     plt

# Importing model classes
from networks.PAMNetwork           import PAMNetwork                # it contains the info about ViT choice, and about the model size

# Importing the dataset class
from RegistrationDataset          import RegistrationDataSet

# Importing helper functions
from utils.utils                  import create_directory
from utils.utils_torch            import weights_init, EarlyStopping
from utils.utils_torch            import draw_curve
from metrics.LossPam              import *
from general_config               import args_pam_fts_sit


def read_train_data():
    path_input= args_pam_fts_sit.train_folder
    path      = Path(path_input)
    filenames = list(path.glob('*.nrrd'))
    data_index= []

    for f in filenames:
        data_index.append(int(str(f).split('/')[5].split('_')[0]))

    train_data = list(zip(data_index, filenames))
    train_data = pd.DataFrame(train_data, columns=['tcia_idx', 'dicom_path'])

    return train_data


def make_directory_to_save_chkp():
    dir_name = args_pam_fts_sit.checkpoints_folder
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    #create_directory(dir_name)


def cuda_seeds():
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark    = False

def save_table(table_name, fixed_img, moving_img, w0_img, w1_img, t1_img):
    table = wandb.Table(columns=['Fixed Image', 'Moving Image', 'Affine Reg. Image', 'Deformation Reg. Image', 'Deformation Field'], allow_mixed_types = True)
    
    saving_examples_folder = args_pam_fts_sit.checkpoints_folder + 'images_each_epoch_wandb/'
    
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

def model_init():
    # Network Definitions
    pam_net = PAMNetwork()

    # Init weights for Generator and Discriminator
    pam_net.apply(weights_init)

    # GPU Computation
    device = torch.device('cuda:0')
    pam_net.to(device)

    return pam_net, device

def init_loss_functions():
    
    disc_loss   = nn.BCELoss()  # discriminator Loss
    l2_loss     = nn.MSELoss()  # Feature matching loss: fts_loss
    CC_loss     = Cross_Correlation_Loss() 
    TV_loss = Energy_Loss()

    return disc_loss, l2_loss, CC_loss, TV_loss


def get_optimizers(pam_net):
    pam_optimizer = torch.optim.Adam(pam_net.parameters(), lr = 3e-4, betas=(0.5, 0.999))

    return pam_optimizer


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


def training(pam_net, device, CC_loss, TV_loss, pam_optimizer,
             train_dataloader, valid_dataloader):
    print ('Starting the Training :D')

    epoch        = 1
    n_epochs     = 10001
    train_losses = []
    valid_losses = []
    alpha_value  = 0.01
    beta_value   = 0.1

    # wandb Initialization
    wandb.init(project=args_pam_fts_sit.wb_project_name, entity='pam_ablation_study')

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=7, verbose=True, path=args_pam_fts_sit.checkpoints_folder + 'PAM_' + args_pam_fts_sit.wb_project_name + '_earlystopping.pth')

    # Saving model inputs and hyperparameters
    config          = wandb.config
    wandb.watch(pam_net)

    it_train_counter = 0
    it_valid_counter = 0
    train_flag = True


    for epoch in range(epoch, n_epochs):

        loss_pam_train    = 0
        loss_pam_valid    = 0       

        pam_net.train()
        
        with tqdm(total=len(train_dataloader)) as pbar:
            
            for i, (x_1, x_2) in enumerate(train_dataloader):

                train_flag = True

                # send to device (GPU or CPU)
                fixed  = x_1.to(device)
                moving = x_2.to(device)

                # zero-grad the parameters
                pam_optimizer.zero_grad()

                # Forward pass generator (1)
                t_0, w_0, t_1, w_1 = pam_net(fixed, moving)

                # Computing the Loss
                # Affine network loss
                cc_affine_loss           = CC_loss.pearson_correlation(fixed, w_0)
                tv_affine_loss           = TV_loss.energy_loss(t_0)               

                # Deformation network loss
                cc_elastic_loss          = CC_loss.pearson_correlation(fixed, w_1)
                tv_elastic_loss          = TV_loss.energy_loss(t_1)
                
                # PAM loss
                loss = (cc_affine_loss + alpha_value*tv_affine_loss) + (cc_elastic_loss + beta_value*tv_elastic_loss)
                loss_pam_train += loss.item()

                # one backward pass
                loss.backward()

                # update the parameters
                pam_optimizer.step()            

                # Display in wandb
                # ========
                wandb.log({'Iteration': it_train_counter,
                           'Train: Similarity Affine loss': cc_affine_loss.item(),
                           'Train: Total Variation Affine loss': alpha_value * tv_affine_loss.item(),
                           'Train: Similarity Elastic loss': cc_elastic_loss.item(),
                           'Train: Total Variation Elastic loss': beta_value * tv_elastic_loss.item(),
                           'Train: Total loss': loss.item()                       
                          })
                pbar.update(1)

                if train_flag:
                    it_train_counter += 1


            with torch.no_grad():

                train_flag = False

                pam_net.eval()

                for i, (x_1, x_2) in enumerate(valid_dataloader):
                    # send to device (GPU or CPU)
                    fixed  = x_1.to(device)
                    moving = x_2.to(device)

                    # Forward pass generator (1)
                    t_0, w_0, t_1, w_1 = pam_net(fixed, moving)

                    # Computing the Loss
                    # Affine network loss
                    cc_affine_loss           = CC_loss.pearson_correlation(fixed, w_0)
                    tv_affine_loss           = TV_loss.energy_loss(t_0)

                    """sim_af_loss, reg_af_loss = total_loss(fixed, w_0, w_0)
                    total_affine             = sim_af_loss + alpha_value * reg_af_loss"""

                    # Deformation network loss
                    cc_elastic_loss          = CC_loss.pearson_correlation(fixed, w_1)
                    tv_elastic_loss          = TV_loss.energy_loss(t_1)

                    """sim_df_loss, reg_df_loss = total_loss(fixed, w_1, t_1)
                    total_elastic            = sim_df_loss + beta_value * reg_df_loss"""                 

                    # PAM loss
                    loss = (cc_affine_loss + alpha_value*tv_affine_loss) + (cc_elastic_loss + beta_value*tv_elastic_loss)
                    loss_pam_valid += loss.item()
                

                    # Display in tensorboard
                    # ========

                    wandb.log({'iteration': it_valid_counter, 
                               'Valid: Similarity Affine loss': cc_affine_loss.item(),
                               'Valid: Total Variation Affine loss': alpha_value * tv_affine_loss.item(),
                               'Valid: Similarity Elastic loss': cc_elastic_loss.item(),
                               'Valid: Total Variation Elastic loss': beta_value * tv_elastic_loss.item(),
                               'Valid: Total loss': loss.item()
                              })

                    if not train_flag:
                        it_valid_counter += 1


            # Compute the loss per epoch
            loss_pam_train     /= len(train_dataloader)          
            train_losses.append(loss_pam_train)

            # Save checkpoints
            if epoch % 10 == 0:
                name_pam = 'PAM_' + args_pam_fts_sit.wb_project_name + '_' + str(epoch) + '.pth'
                torch.save(pam_net.state_dict(), os.path.join(args_pam_fts_sit.checkpoints_folder, name_pam))
                print('Saving model')

            # Compute the loss per epoch            
            loss_pam_valid     /= len(valid_dataloader)
            valid_losses.append(loss_pam_valid)


            # Display in tensorboard
            # ========
            wandb.log({'epoch': epoch+1,
                       'Train: Total loss by epoch': loss_pam_train,                       
                       'Valid: Total loss by epoch': loss_pam_valid,
                      })
            #save_table('Training_examples', fixed_tr, moving_tr, w0_tr, w1_tr, t1_tr)
            #save_table('Validation_examples', fixed_val, moving_val, w0_val, w1_val, t1_val)

            early_stopping(loss_pam_valid, pam_net)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Print the train and validation losses
            print("Train epoch : {}/{}, loss_PAM = {:.6f},".format(epoch, n_epochs, loss_pam_train)) # epoch + 1, n_epochs
            print("Valid epoch : {}/{}, loss_PAM = {:.6f},".format(epoch, n_epochs, loss_pam_valid))
        
        #draw_curve(train_losses, valid_losses)


def start_training():   
    make_directory_to_save_chkp()
    cuda_seeds()
    pam_net, device                    = model_init()
    _, _, CC_loss, TV_loss             = init_loss_functions()
    pam_optimizer                      = get_optimizers(pam_net)
    train_dataloader, valid_dataloader = load_dataloader()
    training(pam_net, device, CC_loss, TV_loss, pam_optimizer,
             train_dataloader, valid_dataloader)


def load_model_weights():   
    # Network definition
    pam_net     = PAMNetwork()

    # GPU computation
    device      = torch.device('cuda:0')
    pam_net.to(device)

    # Loading the model weights
    pam_chkpt = args_pam_fts_sit.pam_checkpoint
    pam_net.load_state_dict(torch.load(pam_chkpt))

    return pam_net, device


def start_retraining():    
    make_directory_to_save_chkp()
    cuda_seeds()
    pam_net, device                    = load_model_weights() #model_init()
    _, _, CC_loss, TV_loss             = init_loss_functions()
    pam_optimizer                      = get_optimizers(pam_net)
    train_dataloader, valid_dataloader = load_dataloader()
    training(pam_net, device, CC_loss, TV_loss, pam_optimizer,
             train_dataloader, valid_dataloader)

  
""" TRAINING OF THE MODEL """   
start_training()
#start_retraining()