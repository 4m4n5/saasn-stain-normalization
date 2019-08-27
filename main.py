import os
from argparse import ArgumentParser
import model as md
from utils import create_link, mkdir
import test as tst
import warnings
import torch
warnings.filterwarnings('ignore')
import gen_samples as gen_samples


class Arguments(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


# +
args = {
    'epochs': 3,
    'decay_epoch': 10,
    'batch_size': 16,
    'lr': 0.0002,
    'load_height': 256,
    'load_width': 256,
    'gpu_ids': '0',
    'crop_height': 256,
    'crop_width': 256,
    'lamda': 10,
    'idt_coef': 0.05,
    'ssim_coef': 0.1,
    'training': True,
    'testing': False,
    'results_dir': '/project/DSone/as3ek/data/ganstain/500_one_one/results/',
    'dataset_dir': '/project/DSone/as3ek/data/ganstain/500_one_one/',
    'checkpoint_dir': '/project/DSone/as3ek/data/ganstain/500_one_one/checkpoint/',
    'norm': 'batch',
    'use_dropout': False,
    'ngf': 64,
    'ndf': 64,
    'gen_net': 'unet_256',
    'dis_net': 'n_layers',
    'self_attn': False,
    'spectral': False,
    'log_freq': 30,
    'custom_tag': 'paper',
    'gen_samples': False,
    'specific_samples': True
}

args = Arguments(args)

tag1 = 'noattn'
if args.self_attn:
    tag1 = 'attn'

tag2 = 'nospec'
if args.spectral:
    tag2 = 'spectral'

# Generate paths for checkpoint and results
args.identifier = str(args.gen_net) + '_' + str(args.dis_net) + '_' + str(args.lamda) + '_' \
+ str(args.lr) + '_' + args.norm + '_' + tag1 + '_' + tag2 + '_' + str(args.batch_size) + '_' \
+ str(args.load_height) + '_ssim_coef_' + str(args.ssim_coef) + '_' + args.custom_tag

args.checkpoint_path = args.checkpoint_dir + args.identifier
args.results_path = args.results_dir + args.identifier


# -

def main(args):
    create_link(args.dataset_dir)
    
    args.gpu_ids = []
    for i in range(torch.cuda.device_count()):
        args.gpu_ids.append(i)
    
    if args.training:
        print('Training')
        model = md.cycleGAN(args)
        model.train(args)
        
    if args.gen_samples:
        print('Generating samples')
        gen_samples.gen_samples(args, 'last')


main(args)
