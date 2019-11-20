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
    'epochs': 10,
    'decay_epoch': 5,
    'batch_size': 1,
    'lr': 0.0002,
    'load_height': 128,
    'load_width': 128,
    'gpu_ids': '0',
    'crop_height': 128,
    'crop_width': 128,
    'alpha': 10, # Cyc loss
    'beta': 5, # Scyc loss
    'gamma': 5, # Dssim loss 
    'delta': 0.5, # Identity
    'training': True,
    'testing': True,
    'results_dir': '/project/DSone/as3ek/data/ganstain/CCHMC_vsi_svs/results/',
    'dataset_dir': '/project/DSone/as3ek/data/ganstain/CCHMC_vsi_svs/',
    'checkpoint_dir': '/project/DSone/as3ek/data/ganstain/CCHMC_vsi_svs/checkpoint/',
    'norm': 'batch',
    'use_dropout': False,
    'ngf': 64,
    'ndf': 64,
    'gen_net': 'unet_128',
    'dis_net': 'n_layers',
    'self_attn': True,
    'spectral': True,
    'log_freq': 50,
    'custom_tag': '',
    'gen_samples': True,
    'specific_samples': False
}

args = Arguments(args)

tag1 = 'noattn'
if args.self_attn:
    tag1 = 'attn'

tag2 = 'nospec'
if args.spectral:
    tag2 = 'spectral'

# Generate paths for checkpoint and results
args.identifier = str(args.gen_net) + '_' + str(args.dis_net) + '_' \
+ str(args.lr) + '_' + args.norm + '_' + tag1 + '_' + tag2 + '_' + str(args.batch_size) + '_' \
+ str(args.load_height) + '_coefs_' + str(args.alpha) + '_' + str(args.beta) + '_' + str(args.gamma) + '_'\
+ str(args.delta) + '_' + args.custom_tag

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


