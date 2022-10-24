import argparse

localizer = argparse.ArgumentParser()
localizer.add_argument('--inp_tcia_path', type=str,
                       default='../../../../../DATA/laura/tcia_abdomen/tcia_dataset_2022.csv',
                       help   ='folder that contains the original file of the tcia dataset')
localizer.add_argument('--inp_cts_path', type=str,
                       default="/IMMUNOTEAM/CancerImagingArchive_20220328/CT/",
                       help   ="folder that contains all the CTs to be processed")
localizer.add_argument('--path_to_save_proc_cts', type=str,
                       default="../../../../../DATA/laura/tcia_abdomen/",
                       help   ="folder to save all the processed CTs (thorax, abdomen or both)")
localizer.add_argument('--proc_tcia_file_name', type=str,
                       default="tcia_20220328_abdomen.csv",
                       help   ="name of the new tcia file (processed)")
localizer.add_argument('--non_proc_tcia_file_name', type=str,
                       default="unprocessed_tcia_20220328_abdomen.csv",
                       help   ="name of the new tcia file (processed)")
localizer.add_argument('--root_path_to_add', type=str,
                       default="../../../../..",
                       help   ="This path is added to the csv in order to read the CTs from the original folder")
args_localizer = localizer.parse_args()
