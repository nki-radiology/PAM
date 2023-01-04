# Prognostic AI-monitoring

This repository contains the code of our research on prognostic AI-monitoring: a prototype for automatic response evaluation to treatment of cancer patients with advanced disease based on deep learning image-to-image registration. 

:construction: This research is still in its preliminary phase, further development and validation is warrant before clinical use.  



## 1. Requirements

- Virtual environment

        $ conda create --name pytorch
        $ conda activate pytorch

- Installing packages inside the virtual environment

        $ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
        $ conda install -c simpleitk simpleitk
        $ conda install -c conda-forge nibabel
        $ conda install -c conda-forge tqdm
        $ conda install -c conda-forge matplotlib
        $ conda install -c anaconda scikit-learn
        $ conda install -c conda-forge scikit-learn-intelex
        $ conda install -c anaconda pandas
        $ conda install -c conda-forge glob2
        $ pip install torchsummary



## 2. Preprocessing Data

Considering there is a trained model to support the preprocessing stage, this model was used. 
Follow the steps to use it and to have your preprocessed data properly.


### 2.1 Requirements

- Virtual environment

        $ conda create --name tf-1.12
        $ conda activate tf-1.12

- Installing packages inside the virtual environment

        $ conda install -c anaconda tensorflow-gpu==1.12
        $ conda install -c anaconda scikit-learn
        $ conda install -c anaconda pandas
        $ conda install -c simpleitk simpleitk
        $ conda install -c conda-forge pydicom
        $ conda install -c conda-forge keras
        $ conda install -c conda-forge nibabel
        $ conda install -c conda-forge tqdm
        $ conda install -c anaconda pillow
        $ conda install -c conda-forge matplotlib


### 2.2 Preprocessing Stage using TCIA Dataset

Before performing the execution of this step, make sure that the config file `config.py` has the right information to just run the code with the file name.

- Testing the localizer model
    
    To see if the localizer is working properly (you can skip this step), open the Localizer.py file and uncomment the main function, and then run:

        $ python Localizer.py
    
    By default, it is using the GPU 0. If you want to change it, modify line 21 ` os.environ["CUDA_VISIBLE_DEVICES"]="0" `


- Preprocessing data

    To load the data considering abdomen or thorax, among other transformations:

        $ python Preprocessing.py


- Format data

    To format the data in different folders for training and testing stages:

        $ python FormatData.py   




## Publications

Stefano Trebeschi, Zuhir Bodalal, Thierry N. Boellaard,  Teresa M. Tareco Bucho, Silvia G. Drago, Ieva Kurilova, Adriana M. Calin-Vainak,  Andrea Delli Pizzi, Mirte Muller, Karlijn Hummelink, Koen J. Hartemink, Thi Dan Linh Nguyen-Kim,  Egbert F. Smit,  Hugo J. Aerts and  Regina G. Beets-Tan; _Prognostic value of deep learning mediated treatment monitoring in lung cancer patients receiving immunotherapy_, Frontiers in Oncology, Cancer Imaging and Imaging directed Interventions, 2021 doi: 10.3389/fonc.2021.609054 [(it's open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.609054)

Stefano Trebeschi, Zuhir Bodalal, Nick van Dijk, Thierry N. Boellaard, Paul Apfaltrer, Teresa M. Tareco Bucho, Thi Dan Linh Nguyen-Kim, Michiel S. van der Heijden, Hugo J. W. L. Aerts and Regina G. H. Beets-Tan; _Development of a Prognostic AI-Monitor for Metastatic Urothelial Cancer Patients Receiving Immunotherapy_, Frontiers in Oncology, Genitourinary Oncology, 2021 doi: 10.3389/fonc.2021.637804 [(it's also open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.637804)



