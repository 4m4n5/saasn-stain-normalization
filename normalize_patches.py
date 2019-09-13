# +
import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis
import kornia
import pandas as pd
import warnings

import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
import random
# -

warnings.filterwarnings('ignore')


class Arguments(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


# +
args = {
    'epochs': 30,
    'decay_epoch': 25,
    'batch_size': 16,
    'lr': 0.0002,
    'load_height': 128,
    'load_width': 128,
    'gpu_ids': '0',
    'crop_height': 128,
    'crop_width': 128,
    'alpha': 5, # Cyc loss
    'beta': 5, # Scyc loss
    'gamma': 2, # Dssim loss 
    'delta': 1, # Identity
    'training': True,
    'testing': True,
    'results_dir': '/project/DSone/as3ek/data/ganstain/1000_SEEM_Cinn/results/',
    'dataset_dir': '/project/DSone/as3ek/data/ganstain/1000_SEEM_Cinn/',
    'checkpoint_dir': '/project/DSone/as3ek/data/ganstain/1000_SEEM_Cinn/checkpoint/',
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


# SOURCE AND TARGET FOLDERS
source_path = '/project/DSone/as3ek/data/patches/1000/unnorm_seem_cinn/train/EE/'
target_path = '/project/DSone/as3ek/data/patches/1000/gannorm_seem_cinn/train/EE/'
train_valid_split = 0.8
size = 256
one_direction = True # If this is false. a -> b -> a will happen. Edit code for otherwise.
gen_name = 'Gba' # Gba to generate b given a, i.e., a -> b

if not os.path.exists(target_path):
    os.makedirs(target_path)

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

args.gpu_ids = []
for i in range(torch.cuda.device_count()):
    args.gpu_ids.append(i)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -

if one_direction:
    G = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
else:
    Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                        use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)

ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_path))
if one_direction:
    G.load_state_dict(ckpt[gen_name])
    G.eval()
else:
    Gab.load_state_dict(ckpt['Gab'])
    Gba.load_state_dict(ckpt['Gba'])
    Gab.eval()
    Gba.eval()
print('Eval mode')

# +
biopsy_patch_no_map = {}
biopsy_target_map = {}

transform = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

for i, patch_name in enumerate(os.listdir(source_path)):
    # Check if patch should be sent to valid for every patch from new patient
    if patch_name.split('__')[0] not in biopsy_patch_no_map: 
        biopsy_patch_no_map[patch_name.split('__')[0]] = 0
        if random.randint(1, 10) > train_valid_split*10:
            biopsy_target_map[patch_name.split('__')[0]] = 'train'
        else:
            biopsy_target_map[patch_name.split('__')[0]] = 'valid'
    # Keeping track of number of patches per biopsy crop        
    biopsy_patch_no_map[patch_name.split('__')[0]] += 1
    
    img = imread(source_path + patch_name)
    img = imresize(img, (size, size))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    image = transform(img)
    image = image.unsqueeze(0)
    if one_direction:
        out = G(image)
    else:
        out = Gba(image)
        out = Gab(out)
    biopsy_target_path = target_path.replace('train', biopsy_target_map[patch_name.split('__')[0]])
    torchvision.utils.save_image((out + 1)/2, biopsy_target_path + patch_name)
    if i % 1000 == 0:
        print(i)
# -

os.listdir(source_path)[0].split('__')


