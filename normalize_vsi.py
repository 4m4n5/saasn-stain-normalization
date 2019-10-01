# %%

# coding: utf-8

# %%


import javabridge
import bioformats
import tqdm
import numpy as np
import tifffile as tf
import math

import os
import glob
import re
from pandas import DataFrame, Series
from PIL import Image
import timeit
import time
import math
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk

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
import argparse
from scipy.misc import imread, imresize
from PIL import Image
import math

from sklearn.feature_extraction.image import reconstruct_from_patches_2d as reconstruct


# %%


warnings.filterwarnings('ignore')


# %%


class Arguments(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


# %%


args = {
    'epochs': 100,
    'decay_epoch': 60,
    'batch_size': 16,
    'lr': 0.0002,
    'load_height': 128,
    'load_width': 128,
    'gpu_ids': '0',
    'crop_height': 128,
    'crop_width': 128,
    'alpha': 6, # Cyc loss
    'beta': 5, # Scyc loss
    'gamma': 2, # Dssim loss 
    'delta': 0.1, # Identity
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
    'custom_tag': 'zif_cinn',
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
args.identifier = str(args.gen_net) + '_' + str(args.dis_net) + '_' + str(args.lr) + '_' + args.norm + '_' + tag1 + '_' + tag2 + '_' + str(args.batch_size) + '_' + str(args.load_height) + '_coefs_' + str(args.alpha) + '_' + str(args.beta) + '_' + str(args.gamma) + '_'+ str(args.delta) + '_' + args.custom_tag

args.checkpoint_path = args.checkpoint_dir + args.identifier
args.results_path = args.results_dir + args.identifier

args.gpu_ids = []
for i in range(torch.cuda.device_count()):
    args.gpu_ids.append(i)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%


# Parameters
one_direction = True # If this is false. a -> b -> a will happen. Edit code for otherwise.
gen_name = 'Gba' # Gba to generate b given a, i.e., a -> b
PATH = '/project/DSone/biopsy_images/SEEM_New_crops_2/'
patch_size = 1000
resize_to = 256
target = '/scratch/as3ek/misc/gannorm_wsi_seem_vsi/' # for WSI
target_path_unnorm = '/project/DSone/as3ek/data/patches/1000/un_normalized/seem_ee_vsi/' # for unnormalized patches
target_path = '/project/DSone/as3ek/data/patches/1000/gan_normalized/seem_ee_vsi/' # for normalized patches
thresh = 0.50
save_WSI = True
overlap = 0.5 # %-age area


# %%


if one_direction:
    G = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
else:
    Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                        use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)


# %%


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


# %%


javabridge.start_vm(class_path=bioformats.JARS)


# %%


def optical_density(tile):
    tile = tile.astype(np.float64)
    od = -np.log((tile+1)/240)
    return od

def keep_tile(tile_tuple, tile_size, tissue_threshold):
    slide_num, tile = tile_tuple
    if tile.shape[0:2] == (tile_size, tile_size):
        tile_orig = tile
        tile = rgb2gray(tile)
        tile = 1 - tile
        tile = canny(tile)
        tile = binary_closing(tile, disk(10))
        tile = binary_dilation(tile, disk(10))
        tile = binary_fill_holes(tile)
        percentage = tile.mean()
        check1 = percentage >= tissue_threshold
        tile = optical_density(tile_orig)
        beta = 0.15
        tile = np.min(tile, axis=2) >= beta
        tile = binary_closing(tile, disk(2))
        tile = binary_dilation(tile, disk(2))
        tile = binary_fill_holes(tile)
        percentage = tile.mean()
        check2 = percentage >= tissue_threshold

        return check1 and check2
    else:
        return False


# %%


def get_img_paths_vsi(train_paths):
    images = {}
    files = glob.glob(os.path.join(train_paths, '*.vsi'))
    for fl in files:
        flbase = os.path.basename(fl)
        flbase_noext = os.path.splitext(flbase)[0]
        images[flbase_noext] = fl
    return images


# %%


transform = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

files = list(get_img_paths_vsi(PATH).values())
num_files = len(files)

for i, file in enumerate(files):
    image = bioformats.ImageReader(file)
    rescale = resize_to / patch_size
    height, width, c = np.array(image.read(rescale=False)).shape
    new_dims = int(rescale * (width // resize_to) * resize_to) ,     int(rescale * (height // resize_to) * resize_to)
    
    file = file.split('/')[-1]
    
    # Initialize x and y coord
    x_cord = 0
    y_cord = 0
    
    if save_WSI:
        joined_image = Image.new('RGB', (new_dims))
    
    while x_cord + patch_size < width - 0:
        while y_cord + patch_size < height - 0:
            patch = Image.fromarray(np.array(image.read(rescale=False, XYWH=(x_cord, y_cord, patch_size, patch_size))))
        
            patch = patch.convert('RGB')
            patch = imresize(patch, (resize_to, resize_to))
            
            # Check if we should keep patch
            if keep_tile((0, patch), resize_to, thresh) == False:
                y_cord = int(y_cord + (1 - overlap) * patch_size)
                continue
                
            patch = patch.transpose(2, 0, 1)
            patch = patch / 255.
            patch = torch.FloatTensor(patch).to(device)
            patch = transform(patch)
            patch = patch.unsqueeze(0)
            
            # Save unnormalized patch
            target_folder = target_path_unnorm
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            filename = target_folder + file.split('.')[0] + '__' + str(x_cord) + '_' + str(y_cord) + '.jpg'
            torchvision.utils.save_image((patch + 1)/2, filename)
            
            if one_direction:
                out = G(patch)
            else:
                out = Gba(patch)
                out = Gab(out)
            
            # Save normalized patch
            target_folder = target_path
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            filename = target_folder + file.split('.')[0] + '__' + str(x_cord) + '_' + str(y_cord) + '.jpg'
            torchvision.utils.save_image((out + 1)/2, filename)
            
            if save_WSI:
                out = (out + 1) / 2
                # this converts it from GPU to CPU and selects first image
                img = out.detach().cpu().numpy()[0]
                #convert image back to Height,Width,Channels
                img = np.transpose(img, (1,2,0))
                patch_join = Image.fromarray(np.uint8(img*255))
                joined_image.paste(patch_join, (int(x_cord*rescale), int(y_cord*rescale)))
            
            # Taking care of overlap
            y_cord = int(y_cord + (1 - overlap) * patch_size)
        
        # Taking care of overlap
        x_cord = int(x_cord + (1 - overlap) * patch_size)
        y_cord = 0
    
    print(str(i + 1) + '/' + str(num_files) + ' Complete!')
    if save_WSI:
        if not os.path.exists(target):
            os.makedirs(target)
        joined_image.save(target + file.split('.')[0] + '.png')


# %%


Image.fromarray(np.array(image.read(z=0, rescale=False, XYWH=(2000, 4000, patch_size, patch_size))))


# %%


image.


# %%


image.read(rescale=False, XYWH=(16000, 2000, patch_size, patch_size))


# %%


project(/DSone/as3ek/data/ganstain/1000_SEEM_Cinn/checkpoint/unet_128_n_layers_0.0002_batch_attn_spectral_16_128_coefs_6_5_2_0.1_zif_cinn/latest.ckpt)

