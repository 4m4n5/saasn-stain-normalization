# %%
import tqdm
import numpy as np
import tifffile as tf
import math

import openslide as opsl
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

import warnings
import staintools

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from PIL import Image
import math

from sklearn.feature_extraction.image import reconstruct_from_patches_2d as reconstruct


# %%
warnings.filterwarnings('ignore')

# %%
# Parameters
PATH = '/project/DSone/biopsy_images/chrc_data_case_preserved/train/EE/1.svs'
patch_size = 1000
resize_to = 1000
target = '/scratch/as3ek/temp/cvpr/norm_WSI/' # for WSI
target_path_unnorm = '/scratch/as3ek/temp/cvpr/patch/unnorm/' # for unnormalized patches
target_path = '/scratch/as3ek/temp/cvpr/patch/norm/' # for normalized patches
thresh = 0.50
save_WSI = True
overlap = 0.5 # %-age area

target_patch = staintools.read_image("/project/DSone/as3ek/data/ganstain/cvpr/500_2_1/trainB/N14-01_01___15250_2750.jpg")
target_patch = Image.fromarray(target_patch)
target_patch = target_patch.resize((1000, 1000))


# %%
image = opsl.OpenSlide('/project/DSone/biopsy_images/chrc_data_case_preserved/train/EE/1.svs')
image.dimensions


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
files = ['/project/DSone/biopsy_images/chrc_data_case_preserved/train/EE/51.svs']
num_files = len(files)

# Stain normalize
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(np.array(target_patch))

for i, file in enumerate(files):
    image = opsl.OpenSlide(file)
    rescale = resize_to / patch_size
    width, height = image.dimensions
    new_dims = int(rescale * (width // resize_to) * resize_to), int(rescale * (height // resize_to) * resize_to)
    
    file = file.split('/')[-1]
    
    # Initialize x and y coord
    x_cord = 0
    y_cord = 0
    
    if save_WSI:
        joined_image = Image.new('RGB', (new_dims))
    
    while x_cord + patch_size < width:
        while y_cord + patch_size < height:
            patch = image.read_region((x_cord, y_cord), 0, (patch_size, patch_size))
        
            patch = patch.convert('RGB')
            patch = patch.resize((resize_to, resize_to))
            patch = np.array(patch)
            
#             # Check if we should keep patch
#             if keep_tile((0, patch), resize_to, thresh) == False:
#                 y_cord = int(y_cord + (1 - overlap) * patch_size)
#                 continue
                
            # Read data
            
            to_transform = patch
            transformed = normalizer.transform(to_transform)
            
            if save_WSI:
                patch_join = Image.fromarray(transformed)
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
        joined_image.save(target + file.split('.')[0] + '.jpg')


# %%
