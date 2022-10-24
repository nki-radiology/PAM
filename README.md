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


## Publications

Stefano Trebeschi, Zuhir Bodalal, Thierry N. Boellaard,  Teresa M. Tareco Bucho, Silvia G. Drago, Ieva Kurilova, Adriana M. Calin-Vainak,  Andrea Delli Pizzi, Mirte Muller, Karlijn Hummelink, Koen J. Hartemink, Thi Dan Linh Nguyen-Kim,  Egbert F. Smit,  Hugo J. Aerts and  Regina G. Beets-Tan; _Prognostic value of deep learning mediated treatment monitoring in lung cancer patients receiving immunotherapy_, Frontiers in Oncology, Cancer Imaging and Imaging directed Interventions, 2021 doi: 10.3389/fonc.2021.609054 [(it's open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.609054)

Stefano Trebeschi, Zuhir Bodalal, Nick van Dijk, Thierry N. Boellaard, Paul Apfaltrer, Teresa M. Tareco Bucho, Thi Dan Linh Nguyen-Kim, Michiel S. van der Heijden, Hugo J. W. L. Aerts and Regina G. H. Beets-Tan; _Development of a Prognostic AI-Monitor for Metastatic Urothelial Cancer Patients Receiving Immunotherapy_, Frontiers in Oncology, Genitourinary Oncology, 2021 doi: 10.3389/fonc.2021.637804 [(it's also open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.637804)



