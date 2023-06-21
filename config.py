import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--img-dim',    
                    type = tuple, 
                    default = (192, 192, 160), 
                    help = 'Image dimension')

parser.add_argument('--body_part',
                    type = str,
                    default = 'abdomen',
                    help = 'body part to train on')

parser.add_argument('--batch-size',
                    type = int,
                    default = 4,
                    help = 'batch size')

parser.add_argument('--filters',
                    type = list,  
                    default = [8, 16, 32, 64, 128, 256, 512, 1024],      
                    help = 'filters number for each layer')

parser.add_argument('--filters-discriminator',     
                    type = list,  
                    default = [8, 16, 32, 64, 128, 256, 512, 1024],    
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
                    default = '/home/s.trebeschi/abdomen_pairs_only_succeed_nrrd.csv',
                    help = 'csv that contains the testing dataset')

parser.add_argument('--project-folder',      
                    type = str,
                    default = '/projects/split-encoders/',
                    help = 'folder that contains checkpoints and log files')

parser.add_argument('--wandb',   
                    type = str,
                    default = "split-encoders",
                    help = "folder to save the model checkpoints")

PARAMS = parser.parse_args()
