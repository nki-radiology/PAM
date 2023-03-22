#!/bin/bash
#SBATCH --time=3-00:00:00                           # Time limit hrs:min:sec
#SBATCH --job-name=split-encoders                   # Job name
#SBATCH --partition=p6000                           # Partition
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --mem=25G                                   # Memory
#SBATCH --cpus-per-task=5                           # Number of cores
#SBATCH --output=/projects/split-encoders/slurm_%j.log   # Standard output and error log
pwd; hostname; date

# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/s.trebeschi/miniconda3/bin/activate pytorch2

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

# Run your command
python /home/s.trebeschi/PAM/train.py