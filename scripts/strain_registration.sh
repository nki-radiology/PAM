#!/bin/bash
#SBATCH --time=3-00:00:00                           # Time limit hrs:min:sec
#SBATCH --job-name=jumbo                            # Job name
#SBATCH --partition=a100                            # Partition
#SBATCH --qos=a100_qos 
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --mem=32G                                   # Memory
#SBATCH --output=/home/s.trebeschi/log-slurm/slurm_%j.log # Standard output and error log
pwd; hostname; date

# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/s.trebeschi/miniconda3/bin/activate pytorch2

# move data to processing
mkdir /processing/s.trebeschi/tcia_train 2>/dev/null
rsync -avv --info=progress2 /data/groups/beets-tan/l.estacio/data_tcia/train/ /processing/s.trebeschi/tcia_train/

mkdir /processing/s.trebeschi/tcia_train_segmentations 2>/dev/null
rsync -avv --info=progress2 /data/groups/beets-tan/s.trebeschi/tcia_train_segmentations/ /processing/s.trebeschi/tcia_train_segmentations/

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

# Run your command
python /home/s.trebeschi/PAM/train.py --batch-size 6 --registration-only True --wandb registration-only --body-part thorax --filters "16,32,64,128,256,512,1024,1024" --filters-discriminator "16,32,64,128,256,512,1024,1024"