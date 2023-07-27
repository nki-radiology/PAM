import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--img-dim',    
                    type = tuple, 
                    default = (192, 192, 192), 
                    help = 'Image dimension')

parser.add_argument('--body-part',
                    type = str,
                    default = 'thorax',
                    help = 'body part to train on')

parser.add_argument('--batch-size',
                    type = int,
                    default = 2,
                    help = 'batch size')

parser.add_argument('--filters',
                    type = str,  
                    default = "32,64,128,256,512,1024,2048",  
                    #default = [8, 8, 16, 16, 32, 32, 32, 32],      
                    help = 'filters number for each layer')

parser.add_argument('--filters-discriminator',     
                    type = str,  
                    default = "32,64,128,256,512,1024,2048",    
                    #default = [8, 8, 16, 16, 32, 32, 32, 32], 
                    help = 'filters number for each layer')

parser.add_argument('--latent-dim',     
                    type = list,  
                    default = 128,    
                    help = 'latent dimension')

parser.add_argument('--train-folder',      
                    type = str,
                    #default = '/data/groups/beets-tan/l.estacio/data_tcia/train/',
                    default = '/processing/s.trebeschi/tcia_train/',
                    help = 'folder that contains the training dataset')

parser.add_argument('--train-folder-segmentations',
                    type = str,
                    #default = '/data/groups/beets-tan/s.trebeschi/tcia_train_segmentations/',
                    default = '/processing/s.trebeschi/tcia_train_segmentations/',
                    help = 'folder that contains the training dataset segmentations')

parser.add_argument('--test-folder',      
                    type = str,
                    default = '/data/groups/beets-tan/l.estacio/data_tcia/test/',
                    help = 'folder that contains the testing dataset')

parser.add_argument('--inference',
                    type = str,
                    default = '/home/s.trebeschi/data/abdomen_pairs_only_succeed_nrrd.csv',
                    help = 'csv that contains the testing dataset')

parser.add_argument('--project-folder',      
                    type = str,
                    default = '/projects/split-encoders/',
                    help = 'folder that contains checkpoints and log files')

parser.add_argument('--wandb',   
                    type = str,
                    default = "split-encoders",
                    help = "folder to save the model checkpoints")

parser.add_argument('--debug',
                    type = bool,
                    default = False,
                    help = 'debug mode')

parser.add_argument('--registration-only',
                    type = bool,
                    default = True,
                    help = 'registration mode')

PARAMS = parser.parse_args()

PARAMS.filters                  = [int(i) for i in PARAMS.filters.split(',')]
PARAMS.filters_discriminator    = [int(i) for i in PARAMS.filters_discriminator.split(',')]