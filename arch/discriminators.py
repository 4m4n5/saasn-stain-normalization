from torch import nn
from .ops import *
import torch
from torch.nn import functional as F
import functools


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False, self_attn=False, spectral=False):
        super(NLayerDiscriminator, self).__init__()
        # convolution : num input channel -> ndf
        dis_model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # every loop increases the num of channels x2 and reduces the size of image x0.5
        # untill it reaches ndf*8 channels
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            
            if self_attn and ndf*nf_mult >=8:
                self_attn_layer = Self_Attn(ndf*nf_mult)
                dis_model += [conv_norm_lrelu(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=2, 
                                              norm_layer=norm_layer, padding=1, bias=use_bias, 
                                              spectral=spectral), self_attn_layer]
            else:
                dis_model += [conv_norm_lrelu(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=2, 
                                              norm_layer=norm_layer, padding=1, bias=use_bias, 
                                              spectral=spectral)]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1,
                                      norm_layer= norm_layer, padding=1, bias=use_bias, spectral=spectral)]

        dis_model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.dis_model = nn.Sequential(*dis_model)
            
    def forward(self, input):
        return self.dis_model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(PixelDiscriminator, self).__init__()
        dis_model = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)


def define_Dis(input_nc, ndf, netD, n_layers_D=3, norm='batch', gpu_ids=[0], spectral=False, self_attn=False):
    dis_net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    if netD == 'n_layers':
        dis_net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_bias=use_bias, spectral=spectral, self_attn=self_attn)
    elif netD == 'pixel':
        dis_net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_bias=use_bias)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    return init_network(dis_net, gpu_ids)


