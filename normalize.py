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
    'epochs': 5,
    'decay_epoch': 3,
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
    'training': False,
    'testing': False,
    'results_dir': '/project/DSone/as3ek/data/ganstain/500/results/',
    'dataset_dir': '/project/DSone/as3ek/data/ganstain/500/',
    'checkpoint_dir': '/project/DSone/as3ek/data/ganstain/500/checkpoint/',
    'norm': 'batch',
    'use_dropout': False,
    'ngf': 64,
    'ndf': 64,
    'gen_net': 'unet_256',
    'dis_net': 'n_layers',
    'self_attn': False,
    'spectral': False,
    'log_freq': 30,
    'custom_tag': '',
    'gen_samples': True,
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

img = imread('/project/DSone/as3ek/data/ganstain/500_one_one/testB/N14-16_00___4750_1500.jpg')
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

torchvision.utils.save_image(out_gba, '/scratch/as3ek/test.jpg')

torchvision.utils.save_image(out_gab, '/scratch/as3ek/test.jpg')


