import os
import shutil
from   pathlib                 import Path
from   sklearn.model_selection import train_test_split
from   config                  import args_localizer


def verify_path_to_save(path: str):
    if not os.path.exists(path):
        print('Creating folder...')
        os.makedirs(path)
    else:
        print('This folder already exists :)!')


def split_train_test():

    # Dataset Path
    path_input = args_localizer.path_to_save_proc_cts   
    path       = Path(path_input)
    filenames  = list(path.glob('*.nrrd'))

    print("Filenames size: ", len(filenames), " ------ ", str(filenames[0]).split('/')[8])

    # Create folders to save data
    train_path = args_localizer.path_to_save_proc_cts + "train/"
    test_path  = args_localizer.path_to_save_proc_cts + "test/"
    verify_path_to_save(train_path)
    verify_path_to_save(test_path)

    # Random seed
    random_seed = 42

    # Split dataset into training set and testing set
    train_size = 0.8
    inputs_train, inputs_test  = train_test_split(
        filenames, random_state=random_seed, train_size=train_size, shuffle=True
    )

    print("Inputs train: ", len(inputs_train))
    print("Inputs test: ", len(inputs_test))

    for f in inputs_train:
        print(f)
        shutil.move(f.absolute(), train_path)

    for f in inputs_test:
        print("Test: ", f)
        shutil.move(f, test_path)

    print("total: ", len(filenames), " train: ", len(inputs_train), " test: ", len(inputs_test))


# split_train_test()
