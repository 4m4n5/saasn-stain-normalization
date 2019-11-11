# %%
import warnings

# Essentials
import os
import openslide as opsl
import numpy as np

# Torch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Image functions
from PIL import Image as PILImage

# Own code
import utils
from arch import define_Gen

# DL Prediction
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.callbacks.hooks import *
from fastai.callback import *

# Misc
import matplotlib.cm as mpl_color_map
import copy

warnings.filterwarnings('ignore')


# %%
args = {
    'epochs': 50,
    'decay_epoch': 40,
    'batch_size': 4,
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
    'results_dir': '/project/DSone/as3ek/data/ganstain/run2/zif_svs/results/',
    'dataset_dir': '/project/DSone/as3ek/data/ganstain/run2/zif_svs/',
    'checkpoint_dir': '/project/DSone/as3ek/data/ganstain/run2/zif_svs/checkpoint/',
    'norm': 'batch',
    'use_dropout': False,
    'ngf': 64,
    'ndf': 64,
    'gen_net': 'unet_128',
    'dis_net': 'n_layers',
    'self_attn': True,
    'spectral': True,
    'log_freq': 50,
    'custom_tag': 'vsi_svs',
    'gen_samples': False,
    'specific_samples': False
}

args = utils.Arguments(args)

tag1 = 'noattn'
if args.self_attn:
    tag1 = 'attn'

tag2 = 'nospec'
if args.spectral:
    tag2 = 'spectral'

# Generate paths for checkpoint and results
args.identifier = str(args.gen_net) + '_' + str(args.dis_net) + '_' + str(args.lr) + '_' + args.norm + '_' + \
tag1 + '_' + tag2 + '_' + str(args.batch_size) + '_' + str(args.load_height) + '_coefs_' + str(args.alpha) + \
'_' + str(args.beta) + '_' + str(args.gamma) + '_'+ str(args.delta) + '_' + args.custom_tag

args.checkpoint_path = args.checkpoint_dir + args.identifier

args.gpu_ids = []
for i in range(torch.cuda.device_count()):
    args.gpu_ids.append(i)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Parameters
one_direction = True # If this is false. b -> a -> b will happen. Edit code for otherwise.
gen_name = 'Gba' # Gba to generate b given a, i.e., a -> b
PATH = '/project/DSone/biopsy_images/ileum/Crohns_ndpi/B1/'
patch_size = 1000
resize_to = 256
target = '/scratch/as3ek/misc/gannorm_ndpi_test/' # for WSI
target_path_unnorm = '/project/DSone/as3ek/data/patches/1000/un_normalized/cinn_normal_svs/' # for unnormalized patches
target_path = '/project/DSone/as3ek/data/patches/1000/gan_normalized/cinn_normal_svs/' # for normalized patches
thresh = 0
save_WSI = False
overlap = 0 # %-age area


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


transform = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

num_files = len(os.listdir(PATH))

for i, file in enumerate(os.listdir(PATH)):
    image = opsl.OpenSlide(PATH + file)
    rescale = resize_to / patch_size
    new_dims = int(rescale * (image.dimensions[0] // 256) * 256) , int(rescale * (image.dimensions[1] // 256) * 256)
    
    # Initialize x and y coord
    x_cord = 0
    y_cord = 0
    
    if save_WSI:
        joined_image = Image.new('RGB', (new_dims))
    
    while x_cord + patch_size < image.dimensions[0] - 1000:
        while y_cord + patch_size < image.dimensions[1] - 1000:
            patch = image.read_region((x_cord, y_cord), 0, (patch_size, patch_size))
        
            patch = patch.convert('RGB')
            patch = patch.resize(resize_to, resize_to)
            
            # Check if we should keep patch
            if utils.keep_tile((0, patch), resize_to, thresh) == False:
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
                out = Gab(patch)
                out = Gba(out)
            
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

