import os
import argparse
import torch
from   utils           import str2bool
from   disentanglement import Disentanglement

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    net = Disentanglement(args=args)
    if args.train:
        net.train_disentanglement_method()
        print("Training!!!!")
    else:
        print('Under development...!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disentanglement Methods: Wassertein Autoencoder, Beta-VAE')
    
    parser.add_argument('--train',      default=True,       type=str2bool,  help='train or testing')
    parser.add_argument('--seed',       default=42,          type=int,       help='random seed')
    parser.add_argument('--cuda',       default=True,       type=str2bool,  help='enable cuda')
    parser.add_argument('--n_epochs',   default=200,        type=int,       help='maximum training iteration')
    parser.add_argument('--batch_size', default=8,          type=int,       help='batch size')

    parser.add_argument('--model',      default='Beta-VAE', type=str,       help='Wasserstein Autoencoder (WAE) and Beta Variational Autoencoder (Beta-VAE)')
    parser.add_argument('--input_ch',   default=2,          type=int,       help='Number of input channels of the image')
    parser.add_argument('--output_ch',  default=1,          type=int,       help='Number of output channels of the image')
    parser.add_argument('--data_dim',   default=2,          type=int,       help='dimension of the data')
    parser.add_argument('--z_dim',      default=256,         type=int,       help='dimension of the representation z')
    parser.add_argument('--img_size',   default=(256, 256),   type=tuple,     help='dimension of the image. now only (256,256) is supported')
    
    parser.add_argument('--lr',         default=1e-4,       type=float,     help='learning rate')
    parser.add_argument('--beta',   	default=4,          type=float,     help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--beta1',      default=0.9,        type=float,     help='Adam optimizer beta1')
    parser.add_argument('--beta2',      default=0.999,      type=float,     help='Adam optimizer beta2')
    
    parser.add_argument('--dset_dir',   default='/DATA/laura/datasets/chest_xray/train/NORMAL', type=str, help='dataset directory')
    parser.add_argument('--num_workers',default=2,         type=int,       help='dataloader num_workers')
    
    parser.add_argument('--ckpt_dir',   default='/DATA/laura/code/PAM/Beta-VAE/checkpoints', type=str,    help='checkpoint directory')
    parser.add_argument('--results_dir',   default='/DATA/laura/code/PAM/Beta-VAE/images_results/', type=str,    help='Results directory of the weight&biases images')
    
    args = parser.parse_args()
    
    main(args)