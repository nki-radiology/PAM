import pandas as pd

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--log-preprocessing',      type = str, default = None,             help = 'log file of preprocessing')
parser.add_argument('--dataset-followup',       type = str, default = None,             help = 'log file of followup')
parser.add_argument('--output',                 type = str, default = None,             help = 'folder that contains the dataset')
parser.add_argument('--debug' ,                 type = bool, default = False,           help = 'debug mode')

PARAMS                                      = parser.parse_args()

LOG_PREPROCESSING                           = PARAMS.log_preprocessing
DATASET_FOLLOWUP                            = PARAMS.dataset_followup
OUTPUT                                      = PARAMS.output
DEBUG                                       = PARAMS.debug


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


if DEBUG:
    breakpoint()


log_preprocessing       = load_log()
followup_dataset        = load_dataset()


print(f' - [info] preprocessing log has {len(log_preprocessing)} entries')
print(f' - [info] followup dataset has {len(followup_dataset)} entries')


followup_dataset = followup_dataset.merge(log_preprocessing, left_on = 'baseline', right_on = 'input_image', how = 'left')
followup_dataset = followup_dataset.merge(log_preprocessing, left_on = 'followup', right_on = 'input_image', how = 'left', suffixes=('_baseline', '_followup'))


followup_dataset.rename({
    'baseline': 'original_image_baseline',
    'followup': 'original_image_followup',
}, inplace=True, axis=1)

followup_dataset.rename({
    'output_image_baseline' : 'baseline',
    'output_image_followup' : 'followup',
}, inplace=True, axis=1)


followup_dataset.to_csv(OUTPUT, index = False)
