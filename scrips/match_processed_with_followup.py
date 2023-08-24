import pandas as pd

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--log-preprocessing',      type = str, default = None,             help = 'log file of preprocessing')
parser.add_argument('--dataset-followup',       type = str, default = None,             help = 'log file of followup')

PARAMS                                      = parser.parse_args()

LOG_PREPROCESSING                           = PARAMS.log_preprocessing
DATASET_FOLLOWUP                            = PARAMS.dataset_followup


def load_log():
    df = pd.read_csv(LOG_PREPROCESSING)
    df = df[['input_image', 'input_mask', 'output_image', 'output_mask']]
    df = df.dropna()
    df = df.reset_index(drop = True)
    return df


def load_dataset():
    df = pd.read_csv(DATASET_FOLLOWUP)
    df = df[['baseline', 'followup']]
    df = df.dropna()
    df = df.reset_index(drop = True)
    return df

log_preprocessing       = load_log()
followup_dataset        = load_dataset()

print(f' - [info] preprocessing log has {len(log_preprocessing)} entries')
print(f' - [info] followup dataset has {len(followup_dataset)} entries')


followup_dataset.merge(log_preprocessing, left_on = 'baseline', right_on = 'input_image', how = 'left', suffixes=('', '_baseline'), inplace = True)
followup_dataset.merge(log_preprocessing, left_on = 'followup', right_on = 'input_image', how = 'left', suffixes=('', '_followup'), inplace = True)

followup_dataset.to_csv('followup_dataset_preprocessed.csv', index = False)
