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


def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride=1, padding=0, 
                    norm_layer = nn.BatchNorm2d, bias=False):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias), 
                         norm_layer(out_dim), nn.LeakyReLU(0.2, True))


def conv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0, 
                    norm_layer = nn.BatchNorm2d, bias=False):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias), 
                         norm_layer(out_dim), nn.ReLU(True))


def dconv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0, output_padding=0,
                    norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding, bias = bias),
                         norm_layer(out_dim), nn.ReLU(True))


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                     conv_norm_relu(dim, dim, kernel_size=3,
                                    norm_layer=norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]
        
        self.res_block = nn.Sequential(*res_block)
        
    def forward(self, x):
        return x + self.res_block(x)


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, 
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == n.InstanceNorm2d
            
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [nn.ReLU(True), upconv, nn.Tanh()]
            model = down + [submodule] + up
        
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
            
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            
            else: 
                model = down + [submodule] + up
                
            self.model = nn.Sequential(*model)
            
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        
        else:
            return torch.cat([x, self.model(x)], 1)


