import argparse
#DONE removed all inputs for channel dimensions and output dimensions, as these should be standards
#DONE changed all image dimensions to be the same for all networks 
# Image variables
parser = argparse.ArgumentParser()
parser.add_argument('--img_dim',     type=tuple, default=(192, 192, 160),         help='Image dimension')
image = parser.parse_args()

# Affine Network
parser = argparse.ArgumentParser()
parser.add_argument('--filters',     type=list,  default=[32, 64, 128, 256, 512], help='filters number for each layer')
affine = parser.parse_args()

# Deformation Network
parser = argparse.ArgumentParser()
parser.add_argument('--filters',     type=list,  default=[32, 64, 128, 256, 512], help='filters number for each layer')
deformation = parser.parse_args()

# Discriminator Network
parser = argparse.ArgumentParser()
parser.add_argument('--filters',     type=list,  default=[8, 16, 32, 64, 128, 256, 512], help='filters number for each layer')
discriminator = parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ Adversarial PAM training variables ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Adversarial PAM using total loss, feature matching loss in the generator, and the alpha, gamma and beta as penalty
# factors. Additionally, it is trained from scratch!
pam_thorax = argparse.ArgumentParser()
pam_thorax.add_argument('--train_folder',      type=str,
                             default='../../../../DATA/laura/tcia/train/',
                             help   ='folder that contains the training dataset')
pam_thorax.add_argument('--checkpoints_folder',   type=str,
                             default="../../../../DATA/laura/tcia/models/pam_adv_fts_sit/",
                             help   ="folder to save the model checkpoints")
args_pam_thorax = pam_thorax.parse_args()


# Abdomen
pam_adv_fts_sit = argparse.ArgumentParser()
pam_adv_fts_sit.add_argument('--train_folder',      type=str,
                             default='/data/groups/beets-tan/l.estacio/data_tcia/train/',
                             help   ='folder that contains the training dataset')
pam_adv_fts_sit.add_argument('--checkpoints_folder',   type=str,
                             default="/projects/disentanglement_methods/temp/PAM/checkpoints_thorax/",
                             help   ="folder to save the model checkpoints")
pam_adv_fts_sit.add_argument('--wb_project_name',   type=str,
                             default="'exp_pam_thorax'",
                             help   ="folder to save the model checkpoints")
pam_adv_fts_sit.add_argument('--pam_checkpoint', type=str,
                             default='/DATA/laura/tcia_abdomen/models/PAMModel_50.pth',
                             help   ="folder that contains the PAM model checkpoints")
pam_adv_fts_sit.add_argument('--dis_checkpoint', type=str,
                             default='/DATA/laura/tcia_abdomen/models/DisModel_50.pth',
                             help   ="folder that contains the discriminator checkpoints")
args_pam_adv_fts_sit = pam_adv_fts_sit.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ Adversarial PAM testing variables ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Feature Matching Loss, penalties and affine output as real image according to Stefano's and Ingmar's Thesis.
# Additionally, loss_d_t.item() < 1e-5 restriction
adv_pam_thorax = argparse.ArgumentParser()
adv_pam_thorax.add_argument('--test_folder', type=str,
                             default='../../../../DATA/laura/tcia/test/',
                             help   ='folder that contains the testing dataset')
adv_pam_thorax.add_argument('--results_folder', type=str,
                             default='../../../../DATA/laura/tcia/results/adv_pam_fts_sit/',
                             help   ="folder to save the visual results")
adv_pam_thorax.add_argument('--pam_checkpoint', type=str,
                             default='../../../../DATA/laura/tcia/models/pam_adv_fts_sit/PAMModel_10.pth',
                             help   ="folder that contains the PAM model checkpoints")
adv_pam_thorax.add_argument('--dis_checkpoint', type=str,
                             default='../../../../DATA/laura/tcia/models/pam_adv_fts_sit/DisModel_10.pth',
                             help   ="folder that contains the discriminator checkpoints")
arg_adv_pam_thorax = adv_pam_thorax.parse_args()


# Abdomen
adv_pam_fts_sit = argparse.ArgumentParser()
adv_pam_fts_sit.add_argument('--test_folder', type=str,
                             default='../../../../DATA/laura/tcia_abdomen/test/',
                             help   ='folder that contains the testing dataset')
adv_pam_fts_sit.add_argument('--results_folder', type=str,
                             default='../../../../DATA/laura/tcia_abdomen/results/',
                             help   ="folder to save the visual results")
adv_pam_fts_sit.add_argument('--pam_checkpoint', type=str,
                             default='../../../../../DATA/laura/tcia_abdomen/models/PAMModel_50.pth',
                             help   ="folder that contains the PAM model checkpoints")
adv_pam_fts_sit.add_argument('--dis_checkpoint', type=str,
                             default='../../../../../DATA/laura/tcia_abdomen/models/DisModel_50.pth',
                             help   ="folder that contains the discriminator checkpoints")
arg_adv_pam_fts_sit = adv_pam_fts_sit.parse_args()