# Prognostic AI-monitoring

This repository contains the code of our research on prognostic AI-monitoring: a prototype for automatic response evaluation to treatment of cancer patients with advanced disease based on deep learning image-to-image registration. 

:construction: This research is still in its preliminary phase, further development and validation is warrant before clinical use.  

## Graphical Abstract



![pam](figures/pam.jpg)

## Installing requirements using Anaconda

- Virtual environment

          $ conda create --name pytorch
          $ conda activate pytorch

- Installing Pytorch inside the virtual environment

          $ conda install -c conda-forge pytorch-gpu 

The Pytorch version (as default) is 1.10.1. You can verify the installed version using>
          $ python
          >>> import torch
          >>> print(torch.__version__)
          >>> exit()

- Installing additional libraries inside the virtual environment

          $ conda install -c anaconda pandas
          $ conda install -c simpleitk simpleitk
          $ conda install -c conda-forge nibabel
          $ conda install -c conda-forge tqdm
          $ conda install -c anaconda pillow
          $ conda install -c conda-forge matplotlib
          $ conda install -c anaconda scikit-learn


Finally, you can check all the installed libraries in your virtual environment using:

          $ conda list


## PAM Execution

### Data Processing



## Citing PAM

If you use this repository or would like to refer the papers, please use the following BibTeX entry:

          @article{trebeschi2021prognostic,
            title={Prognostic Value of Deep Learning-Mediated Treatment Monitoring in Lung Cancer Patients Receiving Immunotherapy},
            author={Trebeschi, Stefano and Bodalal, Zuhir and Boellaard, Thierry N and Bucho, Teresa M Tareco and Drago, Silvia G and Kurilova, Ieva and Calin-Vainak, Adriana M and Pizzi, Andrea Delli and Muller, Mirte and Hummelink, Karlijn and others},
            journal={Frontiers in oncology},
            volume={11},
            year={2021},
            publisher={Frontiers Media SA}
          }


          @article{trebeschi2021development,
            title={Development of a prognostic AI-monitor for metastatic urothelial cancer patients receiving immunotherapy},
            author={Trebeschi, Stefano and Bodalal, Zuhir and Van Dijk, Nick and Boellaard, Thierry N and Apfaltrer, Paul and Bucho, Teresa M Tareco and Nguyen-Kim, Thi Dan Linh and van der Heijden, Michiel S and Aerts, Hugo JWL and Beets-Tan, Regina GH},
            journal={Frontiers in Oncology},
            volume={11},
            year={2021},
            publisher={Frontiers Media SA}
          }

## Publications

Stefano Trebeschi, Zuhir Bodalal, Thierry N. Boellaard,  Teresa M. Tareco Bucho, Silvia G. Drago, Ieva Kurilova, Adriana M. Calin-Vainak,  Andrea Delli Pizzi, Mirte Muller, Karlijn Hummelink, Koen J. Hartemink, Thi Dan Linh Nguyen-Kim,  Egbert F. Smit,  Hugo J. Aerts and  Regina G. Beets-Tan; _Prognostic value of deep learning mediated treatment monitoring in lung cancer patients receiving immunotherapy_, Frontiers in Oncology, Cancer Imaging and Imaging directed Interventions, 2021 doi: 10.3389/fonc.2021.609054 [(it's open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.609054)

Stefano Trebeschi, Zuhir Bodalal, Nick van Dijk, Thierry N. Boellaard, Paul Apfaltrer, Teresa M. Tareco Bucho, Thi Dan Linh Nguyen-Kim, Michiel S. van der Heijden, Hugo J. W. L. Aerts and Regina G. H. Beets-Tan; _Development of a Prognostic AI-Monitor for Metastatic Urothelial Cancer Patients Receiving Immunotherapy_, Frontiers in Oncology, Genitourinary Oncology, 2021 doi: 10.3389/fonc.2021.637804 [(it's also open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.637804)



