import argparse

# ------------------------------------------ General choices of the model -----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--adversarial_choice', type=str, default='no',               help='yes=Adversarial, no=no_Adversarial')
parser.add_argument('--ViT_choice',         type=str, default='no',               help='yes=ViT_PAM,     no=PAM')
parser.add_argument('--size_choice',        type=str, default='big',              help='big, small or big_noskip')
parser.add_argument("-f", "--fff",               default="1",                     help="a dummy argument to fool ipython")
general_choices = parser.parse_args()

# These choices define the model to train and the folders where to save checkpoints:
if general_choices.adversarial_choice == 'no' and general_choices.ViT == 'no':
    experiment_folder = '/projects/pam_valerio/results_Experiment_1'
if general_choices.adversarial_choice == 'yes' and general_choices.ViT == 'no':
    experiment_folder = '/projects/pam_valerio/results_Experiment_2'
if general_choices.adversarial_choice == 'no' and general_choices.ViT == 'yes':
    experiment_folder = '/projects/pam_valerio/results_Experiment_3'
if general_choices.adversarial_choice == 'yes' and general_choices.ViT == 'yes':
    experiment_folder = '/projects/pam_valerio/results_Experiment_4'

if general_choices.size_choice == 'big':
    model_folder = experiment_folder + '/model/model_big/'
    testing_folder = experiment_folder + '/results_testing/model_big/'
    def_filters = [16, 32, 64, 128, 256]
if general_choices.size_choice == 'big_noskip':
    model_folder = experiment_folder + '/model/model_big_noskip/'
    testing_folder = experiment_folder + '/results_testing/model_big_noskip/'
    def_filters = [16, 32, 64, 128, 256]
if general_choices.size_choice == 'small':
    model_folder = experiment_folder + '/model/model_small/'
    testing_folder = experiment_folder + '/results_testing/model_small/'
    def_filters = [4, 8, 16, 32, 64]


# ------------------------------------------ PAM Network Variables ------------------------------------------
# Affine Network
parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int,   default=2,                       help='Input channels')
parser.add_argument('--img_dim',     type=tuple, default=(192, 192, 300),         help='Image dimension')
parser.add_argument("-f", "--fff",               default="1",                     help="a dummy argument to fool ipython")
affine = parser.parse_args()

# Deformation Network

if general_choices.size_choice == 'big_noskip':
    skip_choice = 'no'
else:
    skip_choice = 'yes'

parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int,   default=2,                       help='Input channels')
parser.add_argument('--out_channels',type=int,   default=3,                       help='Output channels')
parser.add_argument('--img_dim',     type=tuple, default=(192, 192, 300),         help='Image dimension')
parser.add_argument('--filters',     type=list,  default=def_filters,             help='Filters of the network')     
parser.add_argument("-f", "--fff",               default="1",                     help="a dummy argument to fool ipython")
parser.add_argument('--skip_choice', type=str,   default=skip_choice,             help='yes=UNet, no=autoencoder')
deformation = parser.parse_args()

# ViT sub-Network

if def_filters == [4, 8, 16, 32, 64]:
    embedding = 1024
elif def_filters == [16, 32, 64, 128, 256]:
    embedding = 4096

parser = argparse.ArgumentParser()
parser.add_argument('--emb_size',    type=int,   default=embedding,               help='Embedding size of ViT_PAM')
parser.add_argument('--ViT_layers',  type=int,   default=12,                      help='Number of Transformer blocks in the network' )
parser.add_argument('--ViT_heads',   type=int,   default=16,                      help='Number of heads used in Transformer block')
visiontransformer = parser.parse_args()

# Discriminator Network
parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int,   default=1,                       help='Input channels')
parser.add_argument('--out_features',type=int,   default=1,                       help='Output features')
parser.add_argument('--filters',     type=list,  default=[8, 16, 32, 64, 128, 256, 512],
                    help='filters number for each layer')
parser.add_argument("-f", "--fff",               default="1",                     help="a dummy argument to fool ipython")
discriminator = parser.parse_args()


# ------------------------------------------ PAM training variables ----------------------------------------

# Example of wandb_name : Experiment1_big
wandb_name = str(model_folder.split('/')[3].split('_')[1]) + str(model_folder.split('/')[3].split('_')[2]) + '_' + str(general_choices.size_choice)

pam_fts_sit = argparse.ArgumentParser()
pam_fts_sit.add_argument('--train_folder',      type=str,
                             default="/processing/valerio/dataset/train/",                             
                             help   ='folder that contains the training dataset')
pam_fts_sit.add_argument('--checkpoints_folder',   type=str,
                             default=model_folder,               
                             help   ="folder to save the model checkpoints")
pam_fts_sit.add_argument('--wb_project_name',   type=str,
                             default=wandb_name,  
                             help   ="name of the project on wandb website")
pam_fts_sit.add_argument('--pam_checkpoint', type=str,
                             default= model_folder + 'ViT_PAM_Model_bignoskip_emb4096_320.pth',     # pay attention only if you have to retrain                   
                             help   ="folder that contains the PAM model checkpoints")
pam_fts_sit.add_argument("-f", "--fff",
                             default="1",
                             help="a dummy argument to fool ipython")
args_pam_fts_sit = pam_fts_sit.parse_args()


# ------------------------------------------ PAM testing variables ----------------------------------------

pam_test_fts = argparse.ArgumentParser()
pam_test_fts.add_argument('--test_folder', type=str,
                             default='/processing/valerio/dataset/test/',
                             help   ='folder that contains the testing dataset')
    
pam_test_fts.add_argument('--results_folder', type=str,
                             default=testing_folder,
                             help   ="folder to save the visual results")
pam_test_fts.add_argument('--pam_checkpoint', type=str,
                             default=model_folder+'ViT_PAM_Model_big_emb4096_250.pth',              # put the name of the choosen model to test
                             help   ="folder that contains the PAM model checkpoints")
pam_test_fts.add_argument("-f", "--fff",
                             default="1",
                             help="a dummy argument to fool ipython")

arg_pam_test_fts = pam_test_fts.parse_args()
