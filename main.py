import os
from argparse import ArgumentParser
import model as md
from utils import create_link
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
    'batch_size': 32,
    'lr': 0.0002,
    'load_height': 256,
    'load_width': 256,
    'gpu_ids': '0',
    'crop_height': 256,
    'crop_width': 256,
    'lamda': 10,
    'idt_coef': 0.5,
    'training': True,
    'testing': True,
    'results_dir': '/project/DSone/ss4yd/chrc_data_patches_1000/results/',
    'dataset_dir': '/project/DSone/ss4yd/chrc_data_patches_1000/',
    'checkpoint_dir': '/project/DSone/ss4yd/chrc_data_patches_1000/results/',
    'norm': 'instance',
    'use_dropout': False,
    'ngf': 64,
    'ndf': 64,
    'gen_net': 'unet_256',
    'dis_net': 'n_layers'
}

args = Arguments(args)


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
        
    if args.testing:
        print('Testing')
        tst


main(args)


