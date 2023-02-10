# Prognostic AI-monitoring - Ablation Study

This branch contains the code of the Image Registration module of PAM; running these scripts it's possible to perform all the experiments.

Experiment 1 : baseline model
Experiment 2 : baseline model + adversarial learning
Experiment 3 : ViT-PAM
Experiment 4 : ViT-PAM + adversarial learning

In order to run the chosen experiment, it's important to check the "general_config.py" file: following the instruction in it, it is possible to select the adversarial_choice, the ViT_choice, the size_choice (big, small, big_noskip version of the model). 
The training scripts (both of them) automatically assign the right name to the checkpoints to save (dependent on the choices), so who wants to run the trainings should just care about the config file. 

## Publications

Stefano Trebeschi, Zuhir Bodalal, Thierry N. Boellaard,  Teresa M. Tareco Bucho, Silvia G. Drago, Ieva Kurilova, Adriana M. Calin-Vainak,  Andrea Delli Pizzi, Mirte Muller, Karlijn Hummelink, Koen J. Hartemink, Thi Dan Linh Nguyen-Kim,  Egbert F. Smit,  Hugo J. Aerts and  Regina G. Beets-Tan; _Prognostic value of deep learning mediated treatment monitoring in lung cancer patients receiving immunotherapy_, Frontiers in Oncology, Cancer Imaging and Imaging directed Interventions, 2021 doi: 10.3389/fonc.2021.609054 [(it's open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.609054)

Stefano Trebeschi, Zuhir Bodalal, Nick van Dijk, Thierry N. Boellaard, Paul Apfaltrer, Teresa M. Tareco Bucho, Thi Dan Linh Nguyen-Kim, Michiel S. van der Heijden, Hugo J. W. L. Aerts and Regina G. H. Beets-Tan; _Development of a Prognostic AI-Monitor for Metastatic Urothelial Cancer Patients Receiving Immunotherapy_, Frontiers in Oncology, Genitourinary Oncology, 2021 doi: 10.3389/fonc.2021.637804 [(it's also open access!)](https://www.frontiersin.org/articles/10.3389/fonc.2021.637804)



