import argparse

parser = argparse.ArgumentParser()

# parameters that can be changed from the command line
parser.add_argument('--body-part',          type = str, default = 'thorax',         help = 'body part to train on')

parser.add_argument('--dataset-csv',        type = str, default = None,             help = 'csv file with filepaths on the first column')
parser.add_argument('--dataset-folder',     type = str, default = None,             help = 'folder that contains the dataset')
parser.add_argument('--project-folder',     type = str, default = '/projects/split-encoders/', help = 'folder that contains checkpoints and log files')

parser.add_argument('--module',             type = str, default = 'registration',   help = 'registration, student')
parser.add_argument('--task',               type = str, default = 'train',          help = 'train, debug, inference')
parser.add_argument('--batch-size',         type = int, default = 2,                help = 'batch size')
parser.add_argument('--wandb',              type = str, default = "split-encoders", help = "wanddb project name")

PARAMS                                      = parser.parse_args()

# parameters that are fixed
PARAMS.img_dim                              = [192, 192, 192]            
PARAMS.filters                              = [16, 32, 64, 128, 256, 512, 1024, 2048]
PARAMS.filters_discriminator                = [16, 32, 64, 128, 256, 512, 1024, 2048]
PARAMS.latent_dim                           = 128

if PARAMS.task == 'debug':
    PARAMS.filters                          = [8, 16, 16, 16, 32, 32, 32, 64]
    PARAMS.filters_discriminator            = [8, 16, 16, 16, 32, 32, 32, 64]
    PARAMS.latent_dim                       = 16

# print out parameters
print('#############################################')
print('Parameters:')
print(PARAMS)
print('#############################################')
