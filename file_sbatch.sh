#!/bin/bash
#SBATCH --job-name=exp2_big                        # Job name
#SBATCH --partition=rtx8000                         # Partition
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --mem=20G                                   # Memory
#SBATCH --cpus-per-task=5                          # Number of cores
#SBATCH --time=96:00:00                            # Time limit hrs:min:sec
#SBATCH --output=/projects/pam_valerio/outputjobs/slurm_%j.log   # Standard output and error log
pwd; hostname; date

# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate pytorch2

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

# Run your command
python /projects/pam_valerio/code/PAM/TrainPAMAdversarial.py