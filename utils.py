# -*- coding: utf-8 -*-
# + {}
import copy
import os
import shutil
import numpy as np
import torch

from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
from scipy.ndimage.morphology import binary_fill_holes


# -

# Arguments
class Arguments(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


def process_args(args):
    tag1 = 'noattn'
    if args.self_attn:
        tag1 = 'attn'

    tag2 = 'nospec'
    if args.spectral:
        tag2 = 'spectral'

    # Generate paths for checkpoint and results
    args.identifier = str(args.gen_net) + '_' + str(args.dis_net) + '_' + str(args.lr) + '_' + args.norm + '_'\
    + tag1 + '_' + tag2 + '_' + str(args.batch_size) + '_' + str(args.load_height) + '_coefs_' + str(args.alpha)\
    + '_' + str(args.beta) + '_' + str(args.gamma) + '_'+ str(args.delta) + '_' + args.custom_tag

    args.checkpoint_path = args.checkpoint_dir + args.identifier

    args.gpu_ids = []
    for i in range(torch.cuda.device_count()):
        args.gpu_ids.append(i)
    
    return args


# Make directories
def mkdir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


# Make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


def create_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    dirs['sampleA'] = os.path.join(dataset_dir, 'lsampleA')
    dirs['sampleB'] = os.path.join(dataset_dir, 'lsampleB')
    mkdir(dirs.values())

    for key in dirs:
        try:
            os.remove(os.path.join(dirs[key], 'Link'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
                   os.path.join(dirs[key], 'Link'))

    return dirs


# +
def get_traindata_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    return dirs

def get_testdata_link(dataset_dir):
    dirs = {}
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    return dirs

def get_sampledata_link(dataset_dir):
    dirs = {}
    dirs['sampleA'] = os.path.join(dataset_dir, 'lsampleA')
    dirs['sampleB'] = os.path.join(dataset_dir, 'lsampleB')
    return dirs


# -

# Save checkpoint 
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# Load checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)


def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i=i+1
    print('-----------------------------------------------')


# +
def optical_density(tile):
    tile = tile.astype(np.float64)

    od = -np.log((tile+1)/240)
    return od

def keep_tile(tile_tuple, tile_size, tissue_threshold):
    slide_num, tile = tile_tuple
    if tile.shape[0:2] == (tile_size, tile_size):
        tile_orig = tile
        tile = rgb2gray(tile)
        # 8-bit depth complement, from 1 (dense tissue)
        # to 0 (plain background).
        tile = 1 - tile
        # Canny edge detection with hysteresis thresholding.
        # This returns a binary map of edges, with 1 equal to
        # an edge. The idea is that tissue would be full of
        # edges, while background would not.
        tile = canny(tile)
        # Binary closing, which is a dilation followed by
        # an erosion. This removes small dark spots, which
        # helps remove noise in the background.
        tile = binary_closing(tile, disk(10))
        # Binary dilation, which enlarges bright areas,
        # and shrinks dark areas. This helps fill in holes
        # within regions of tissue.
        tile = binary_dilation(tile, disk(10))
        # Fill remaining holes within regions of tissue.
        tile = binary_fill_holes(tile)
        # Calculate percentage of tissue coverage.
        percentage = tile.mean()
        check1 = percentage >= tissue_threshold

        # Check 2
        # Convert to optical density values
        tile = optical_density(tile_orig)
        # Threshold at beta
        beta = 0.15
        tile = np.min(tile, axis=2) >= beta
        # Apply morphology for same reasons as above.
        tile = binary_closing(tile, disk(2))
        tile = binary_dilation(tile, disk(2))
        tile = binary_fill_holes(tile)
        percentage = tile.mean()
        check2 = percentage >= tissue_threshold

        return check1 and check2
    else:
        return False
