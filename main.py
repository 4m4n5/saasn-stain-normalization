import os
from argparse import ArgumentParser
import model as md
from utils import create_link, mkdir
import test as tst
import warnings
import torch
warnings.filterwarnings('ignore')


class Arguments(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


# +
args = {
    'epochs': 200,
    'decay_epoch': 100,
    'batch_size': 8,
    'lr': 0.0002,
    'load_height': 128,
    'load_width': 128,
    'gpu_ids': '0',
    'crop_height': 128,
    'crop_width': 128,
    'lamda': 10,
    'idt_coef': 0.5,
    'training': True,
    'testing': True,
    'results_dir': '/project/DSone/ss4yd/chrc_data_patches_1000_ke/results/',
    'dataset_dir': '/project/DSone/ss4yd/chrc_data_patches_1000_ke/',
    'checkpoint_dir': '/project/DSone/ss4yd/chrc_data_patches_1000_ke/checkpoint/',
    'norm': 'batch',
    'use_dropout': False,
    'ngf': 64,
    'ndf': 64,
    'gen_net': 'unet_128',
    'dis_net': 'n_layers',
    'self_attn': True,
    'spectral': True
}

args = Arguments(args)

tag1 = ''
if args.self_attn:
    tag1 = 'attn'

tag2 = ''
if args.spectral:
    tag2 = 'spectral'

# Generate paths for checkpoint and results
args.checkpoint_path = args.checkpoint_dir + str(args.gen_net) + '_' + str(args.dis_net) + '_' + str(args.lamda) + '_' + str(args.lr) + '_' + args.norm + '_' + tag1 + '_' + tag2
args.results_path = args.results_dir + str(args.gen_net) + '_' + str(args.dis_net) + '_' + str(args.lamda) + '_' + str(args.lr) + '_' + args.norm + '_' + tag1 + '_' + tag2
mkdir([args.checkpoint_path, args.results_path])


# -

def main(args):
    create_link(args.dataset_dir)
    
    args.gpu_ids = []
    for i in range(torch.cuda.device_count()):
        args.gpu_ids.append(i)
    
    if args.training:
        print('Training')
        model = md.cycleGAN(args)
        print(model)
        model.train(args)
        
    if args.testing:
        print('Testing')
        tst.test(args, 'last')


main(args)
