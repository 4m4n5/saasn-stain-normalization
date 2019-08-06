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


