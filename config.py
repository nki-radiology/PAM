import argparse

parser = argparse.ArgumentParser()

# parameters that can be changed from the command line
parser.add_argument('--body-part',          type = str, default = 'thorax',         help = 'body part to train on')

parser.add_argument('--dataset-folder',     type = str, default = None,             help = 'folder that contains the dataset')
parser.add_argument('--dataset-followup',   type = str, default = None,             help = 'csv that contains the images of a follow-up study')
parser.add_argument('--project-folder',     type = str, default = '/projects/split-encoders/', help = 'folder that contains checkpoints of the model')
parser.add_argument('--output-folder',      type = str, default = None,             help = 'self-explanatory')

parser.add_argument('--module',             type = str, default = 'registration',   help = 'registration, student')
parser.add_argument('--debug',              type = bool, default = False,           help = 'self-explanatory')
parser.add_argument('--batch-size',         type = int, default = 2,                help = 'self-explanatory')
parser.add_argument('--wandb',              type = str, default = "split-encoders", help = "wanddb project name")

parser.add_argument('--keep-network-size',  type = bool, default = False,           help = 'in combo with debug, keep the network size')

PARAMS                                      = parser.parse_args()

# parameters that are fixed
PARAMS.img_dim                              = [192, 192, 192]            
PARAMS.filters                              = [16, 32, 64, 128, 256, 512, 1024, 1024]
PARAMS.filters_discriminator                = [16, 32, 64, 128, 256, 512, 1024, 1024]
PARAMS.latent_dim                           = 128

if PARAMS.debug and not PARAMS.keep_network_size:
    PARAMS.filters                          = [8, 16, 16, 16, 32, 32, 32, 64]
    PARAMS.filters_discriminator            = [8, 16, 16, 16, 32, 32, 32, 64]
    PARAMS.latent_dim                       = 16


# print out parameters
print('#############################################')
print('Parameters:')
print(PARAMS)
print('#############################################')
