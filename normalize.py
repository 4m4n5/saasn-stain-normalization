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
    'delta': 0.1, # Identity
    'training': True,
    'testing': True,
    'results_dir': '/project/DSone/as3ek/data/ganstain/500/results/',
    'dataset_dir': '/project/DSone/as3ek/data/ganstain/500/',
    'checkpoint_dir': '/project/DSone/as3ek/data/ganstain/500/checkpoint/',
    'norm': 'batch',
    'use_dropout': False,
    'ngf': 64,
    'ndf': 64,
    'gen_net': 'unet_128',
    'dis_net': 'n_layers',
    'self_attn': True,
    'spectral': True,
    'log_freq': 50,
    'custom_tag': 'p100',
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

args.gpu_ids = []
for i in range(torch.cuda.device_count()):
    args.gpu_ids.append(i)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -

Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)

ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_path))
Gab.load_state_dict(ckpt['Gab'])
Gba.load_state_dict(ckpt['Gba'])

img = imread('/project/DSone/as3ek/data/ganstain/500/testA/130377_6729_001___3250_4500.jpg')
Gab.eval()
Gba.eval()
print('Eval mode')

img = imresize(img, (256, 256))
img = img.transpose(2, 0, 1)
img = img / 255.
img = torch.FloatTensor(img).to(device)

transform = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image = transform(img)
image = image.unsqueeze(0)

out_gab = Gab(image)

out_gba = Gba(image)

torchvision.utils.save_image((out_gba + 1)/2, '/scratch/as3ek/test.jpg')

torchvision.utils.save_image((out_gab + 1)/2, '/scratch/as3ek/test2.jpg')

gray = kornia.color.RgbToGrayscale()
m = kornia.losses.SSIM(11, 'mean')
ba_ssim = m(gray((image + 1) / 2.0), gray((out_gba + 1) / 2.0))
ab_ssim = m(gray((image + 1) / 2.0), gray((out_gab + 1) / 2.0))


