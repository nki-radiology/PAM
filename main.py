import os
import argparse
import torch
from   utils import str2bool
from   train import Train

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    net = Train(args=args)
    if args.train:
        net.train_disentanglement_method()
        print("Training!!!!")
    else:
        print('Under development...!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disentanglement Methods: Wassertein Autoencoder, Beta-VAE')
    
    parser.add_argument('--train',      default=True,       type=str2bool,  help='train or testing')
    parser.add_argument('--seed',       default=42,         type=int,       help='random seed')
    parser.add_argument('--cuda',       default=True,       type=str2bool,  help='enable cuda')
    parser.add_argument('--num_gpus',   default=1,          type=int,       help='Number of GPUs available. Use 0 for CPU mode.')
    parser.add_argument('--num_workers',default=8,          type=int,       help='dataloader num_workers')
    parser.add_argument('--start_ep',   default=1,          type=int,       help='start training iteration')
    parser.add_argument('--n_epochs',   default=2000,       type=int,       help='maximum training iteration')
    parser.add_argument('--batch_size', default=8,          type=int,       help='batch size')

    parser.add_argument('--model',      default='Beta-VAE', type=str,       help='Wasserstein Autoencoder (WAE) and Beta Variational Autoencoder (Beta-VAE)')
    parser.add_argument('--add_disc',   default=True,      type=bool,      help='Add a discriminator network to the Beta-VAE model')
    parser.add_argument('--input_ch',   default=2,          type=int,       help='Number of input channels of the image')
    parser.add_argument('--input_ch_d', default=1,          type=int,       help='Number of input channels of the image for the discriminator')
    parser.add_argument('--output_ch',  default=3,          type=int,       help='Number of output channels of the image')
    parser.add_argument('--latent_dim', default=512,        type=int,       help='dimension of the representation z')
    parser.add_argument('--group_num',  default=8,          type=int,       help='Group normalization size')
    parser.add_argument('--input_dim',  default=[192, 192, 304],    type=int,     help='dimension of the data')
    parser.add_argument('--filters',    default=[8, 16, 32, 64], type=object,  help='filters to create the Beta-VAE')
    parser.add_argument('--filters_disc',default=[8, 16, 32, 64, 128, 256, 512], type=object,  help='filters to create the discriminator')
    
    parser.add_argument('--alpha_value', default=0.01,      type=float,     help='beta parameter for the penalty loss')
    parser.add_argument('--beta_value',  default=0,         type=float,     help='beta parameter for the KL-term loss')
    parser.add_argument('--beta_value_max', default=4,      type=float,     help='Maximum beta parameter for the KL-term loss')
    parser.add_argument('--gamma_value', default=0.1,       type=float,     help='gamma parameter for for the discriminator (feature matching loss: MSE)')
    parser.add_argument('--lambda_value',default=0.001,     type=float,     help='lambda parameter for the reconstruction loss')
    parser.add_argument('--lr',         default=1e-5,       type=float,     help='learning rate')
    parser.add_argument('--beta1',      default=0.9,        type=float,     help='Adam optimizer beta1')
    parser.add_argument('--beta2',      default=0.999,      type=float,     help='Adam optimizer beta2')
    parser.add_argument('--accum_iter_b',default=8,         type=int,       help='Integer value indicating how many batches we need before updating the network weights')
    
    parser.add_argument('--proj_wae',              default='WAE',                  type=str, help='Project name for weights and biases for the WAe model')
    parser.add_argument('--proj_bvae_vanilla',     default='Beta-VAE',             type=str, help='Project name for weights and biases for the vanilla Beta-VAE')
    parser.add_argument('--proj_bvae_adversarial', default='Beta-VAE-Adversarial', type=str, help='Project name for weights and biases for the adversarial Beta-VAE')
    parser.add_argument('--entity_wb',             default='ljestaciocerquin',     type=str, help='Entity for weights and biases')
    
    parser.add_argument('--dset_dir',   default='/data/groups/beets-tan/l.estacio/data_tcia/train/',                    type=str, help='dataset directory')
    parser.add_argument('--ckpt_dir',   default='/projects/disentanglement_methods/checkpoints/Beta_VAE/adversarial/',  type=str, help='checkpoint directory')
    parser.add_argument('--results_dir',default='/projects/disentanglement_methods/results/Beta_VAE/adversarial/',      type=str, help='Results directory of the weight&biases images')
    
    args = parser.parse_args()
    
    main(args)
    
            