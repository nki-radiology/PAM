# Prognostic AI-monitoring - Ablation Study

This branch contains the code of the Image Registration module of PAM; running these scripts it's possible to perform all the experiments.

Experiment 1 : baseline model

Experiment 2 : baseline model + adversarial learning

Experiment 3 : ViT-PAM

Experiment 4 : ViT-PAM + adversarial learning


In order to run the chosen experiment, it's important to check the "general_config.py" file: following the instruction in it, it is possible to select the adversarial_choice, the ViT_choice, the size_choice (big, small, big_noskip version of the model). 

The training scripts (both of them) automatically assign the right name to the checkpoints to save (dependent on the choices), so who wants to run the trainings should just care about the config file. 

`For Experiments 1 and 3 (not adversarial ones), the "TrainPAMAdversarial.py" script has to be used.`

`For Experiments 2 and 4 (adversarial ones), the "TrainPAM_noAdversarial.py" script has to be used.`

## How to run a training

It's needed a sbatch file, where to set some parameters as the name of the job, the partition, the time-limit ...

Example of sbatch file:

```ruby
#!/bin/bash
#SBATCH --job-name=experiment                       # Job name
#SBATCH --partition=rtx8000                         # Partition
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --mem=60G                                   # Memory
#SBATCH --cpus-per-task=10                          # Number of cores
#SBATCH --time=120:00:00                            # Time limit hrs:min:sec
#SBATCH --output=/projects/pam_valerio/outputjobs/slurm_%j.log   # Standard output and error log
pwd; hostname; date

# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/v.pugliese/miniconda3/bin/activate pytorch2

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

# Run your command
python /projects/pam_valerio/code/PAM/TrainPAM_noAdversarial.py 
```
Who wants to run it should change the virtual environment.

## Requirements

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c simpleitk simpleitk
conda install -c conda-forge nibabel
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib
conda install -c anaconda scikit-learn
conda install -c conda-forge scikit-learn-intelex
conda install -c anaconda pandas
conda install -c conda-forge glob2
pip install torchsummary
conda install -c conda-forge wandb
conda install -c conda-forge einops


## Publications

Stefano Trebeschi, Zuhir Bodalal, Thierry N. Boellaard,  Teresa M. Tareco Bucho, Silvia G. Drago, Ieva Kurilova, Adriana M. Calin-Vainak,  Andrea Delli Pizzi, Mirte Muller, Karlijn Hummelink, Koen J. Hartemink, Thi Dan Linh Nguyen-Kim,  Egbert F. Smit,  Hugo J. Aerts and  Regina G. Beets-Tan; _Prognostic value of deep learning mediated treatment monitoring in lung cancer patients receiving immunotherapy_, Frontiers in Oncology, Cancer Imaging and Imaging directed Interventions, 2021 doi: 10.3389/fonc.2021.609054 [(it's open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.609054)

Stefano Trebeschi, Zuhir Bodalal, Nick van Dijk, Thierry N. Boellaard, Paul Apfaltrer, Teresa M. Tareco Bucho, Thi Dan Linh Nguyen-Kim, Michiel S. van der Heijden, Hugo J. W. L. Aerts and Regina G. H. Beets-Tan; _Development of a Prognostic AI-Monitor for Metastatic Urothelial Cancer Patients Receiving Immunotherapy_, Frontiers in Oncology, Genitourinary Oncology, 2021 doi: 10.3389/fonc.2021.637804 [(it's also open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.637804)



