import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--img-dim',    
                    type = tuple, 
                    default = (192, 192, 160), 
                    help = 'Image dimension')

parser.add_argument('--filters',
                    type = list,  
                    default = [16, 32, 64, 128, 256, 512],      
                    help = 'filters number for each layer')

parser.add_argument('--filters-discriminator',     
                    type = list,  
                    default = [32, 64, 128, 256, 384, 512, 1024],    
                    help = 'filters number for each layer')

parser.add_argument('--train-folder',      
                    type = str,
                    default = '/data/groups/beets-tan/l.estacio/data_tcia/train/',
                    help = 'folder that contains the training dataset')

parser.add_argument('--test-folder',      
                    type = str,
                    default = '/data/groups/beets-tan/l.estacio/data_tcia/test/',
                    help = 'folder that contains the testing dataset')

parser.add_argument('--project-folder',      
                    type = str,
                    default = '/projects/split-encoders/',
                    help = 'folder that contains checkpoints and log files')

parser.add_argument('--wandb',   
                    type = str,
                    default = "split-encoders",
                    help = "folder to save the model checkpoints")




PARAMS = parser.parse_args()
