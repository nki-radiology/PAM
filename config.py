# Arguments for the Affine Transformation Network
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_ch", type=int, default=3,  help="input channels")
parser.add_argument("--in_fs", type=int, default=16, help="input features")
args_at = parser.parse_args()


