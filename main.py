import os
import argparse
import torch
from   utils           import str2bool
from   disentanglement import Disentanglement
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    parser.add_argument('--seed',       default=42,         type=int,       help='random seed')
    parser.add_argument('--cuda',       default=True,       type=str2bool,  help='enable cuda')
    parser.add_argument('--num_gpus',   default=1,          type=int,       help='Number of GPUs available. Use 0 for CPU mode.')
    parser.add_argument('--start_ep',   default=1,          type=int,       help='start training iteration')
    parser.add_argument('--n_epochs',   default=2000,       type=int,       help='maximum training iteration')
    parser.add_argument('--batch_size', default=8,          type=int,       help='batch size')

    parser.add_argument('--model',      default='Beta-VAE', type=str,       help='Wasserstein Autoencoder (WAE) and Beta Variational Autoencoder (Beta-VAE)')
    parser.add_argument('--add_disc',   default=True,       type=bool,      help='Add a discriminator network to the Beta-VAE model')
    parser.add_argument('--input_ch',   default=2,          type=int,       help='Number of input channels of the image')
    parser.add_argument('--input_ch_d', default=1,          type=int,       help='Number of input channels of the image for the discriminator')
    parser.add_argument('--output_ch',  default=3,          type=int,       help='Number of output channels of the image')
    parser.add_argument('--input_dim',  default=[256, 256, 512], type=int,  help='dimension of the data')
    parser.add_argument('--latent_dim', default=512,        type=int,       help='dimension of the representation z')
    parser.add_argument('--group_num',  default=8,          type=int,       help='Group normalization size')
    parser.add_argument('--filters',    default=[32, 64, 128, 256], type=object,  help='dimension of the data')
    
    parser.add_argument('--lr',         default=3e-4,       type=float,     help='learning rate')
    parser.add_argument('--beta',   	default=4,          type=float,     help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--beta1',      default=0.5,        type=float,     help='Adam optimizer beta1')
    parser.add_argument('--beta2',      default=0.999,      type=float,     help='Adam optimizer beta2')
    
    parser.add_argument('--dset_dir',   default='/SHARED/active_Laura/chest_xray/train/NORMAL',  type=str, help='dataset directory')
    parser.add_argument('--num_workers',default=2,          type=int,       help='dataloader num_workers')
    
    parser.add_argument('--ckpt_dir',   default='/DATA/laura/code/PAM/Beta-VAE-adversarial-rot/checkpoints',  type=str, help='checkpoint directory')
    parser.add_argument('--results_dir',default='/DATA/laura/code/PAM/Beta-VAE-adversarial-rot/img_results/', type=str, help='Results directory of the weight&biases images')
    
    args = parser.parse_args()
    
    main(args)