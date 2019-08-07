import functools
import torch
from torch import nn
from ops import *


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        
        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=None, norm_layer=norm_layer, innermost=True)
        
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        
        self.unet_model = unet_block
        
    def forward(self, input):
        return self.unet_model(input)


def ResnetGene
