#!/bin/bash
#SBATCH --time=7-00:00:00                           # Time limit hrs:min:sec
#SBATCH --job-name=belbo                            # Job name
#SBATCH --partition=rtx8000                         # Partition
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --mem=32G                                   # Memory
#SBATCH --output=/home/s.trebeschi/log-slurm/slurm_%j.log # Standard output and error log
pwd; hostname; date

# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/s.trebeschi/miniconda3/bin/activate totseg

# move data to processing
mkdir /processing/s.trebeschi/totseg 2>/dev/null

#mkdir /processing/s.trebeschi/tcia_train_segmentations 2>/dev/null
#rsync -avv --info=progress2 /data/groups/beets-tan/s.trebeschi/tcia_train_segmentations/ /processing/s.trebeschi/tcia_train_segmentations/

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

# Run your command
python /home/s.trebeschi/PAM/totalseg/run.py --input /data/groups/beets-tan/s.trebeschi/QOL_dicoms/ --output /processing/s.trebeschi/totseg 

rsync -avv --info=progress2 /processing/s.trebeschi/totseg/ /data/groups/beets-tan/s.trebeschi/QOL_dicoms/TOTSEG/