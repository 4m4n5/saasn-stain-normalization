import functools
from torch.nn import init
import torch.nn as nn
import torch


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_layer == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_status=False)
    else:
        raise NotImplementedError('norm layer [%s] not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            else:
                raise NotImplementedError('[%s] init type not not found' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            else:
                raise NotImplementedError('[%s] init type not not found' % init_type)
            init.constant(m.bias.data, 0.0)
            
    print('Network initialized with weights sampled from N(0,[%s]).' % gain)
    net.apply(init_func)


def init_network(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


