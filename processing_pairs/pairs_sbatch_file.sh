#!/bin/bash
#SBATCH --time=7-00:00:00                           # Time limit hrs:min:sec
#SBATCH --job-name=Thx3_infoA                       # Job name
#SBATCH --partition=rtx8000                         # Partition
#SBATCH --nodelist=roentgen                         # Node list
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --cpus-per-task=6                           # Number of cores
#SBATCH --output=/projects/disentanglement_methods/outputjobs/pairs/slurm_%j.log   # Standard output and error log
pwd; hostname; date

# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate tensorflow

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

# Run your command
python /projects/disentanglement_methods/PAM/processing_pairs/pairs_format.py
