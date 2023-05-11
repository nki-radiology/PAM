#!/bin/bash
#SBATCH --time=7-00:00:00                           # Time limit hrs:min:sec
#SBATCH --job-name=srvA_pam                         # Job name
#SBATCH --qos=a6000_qos
#SBATCH --partition=a6000                           # Partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/projects/disentanglement_methods/outputjobs/PAM_Survival/slurm_%j.log   # Standard output and error log
pwd; hostname; date

# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate pytorch

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

# Run your command
python /projects/disentanglement_methods/survival_PAM/PAM/main.py