# This script define the asymmetric quantization method


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable


class RoundWithGradient(Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


def SyQuan(x, k, x_min, x_max):
    """
    Symmetric k-bit quantization
    """

    # prevent 0 in denominator
    delta = (2**k-1) / torch.clamp((x_max - x_min), min=1e-10)

    b = delta * x_min

    b += (2**(k-1))
    
    # round zero point(zero point should also be interger)
    b = b.round()
    
    # for convolution
    if len(x.shape) == 4:
        delta = delta.view(-1, 1, 1, 1)
        b = b.view(-1, 1, 1, 1)
    elif len(x.shape) == 2:
        delta = delta.view(-1, 1)
        b = b.view(-1, 1)
        
    # quan
    x = delta * x - b

    
    x = RoundWithGradient.apply(x)    

    limit = (2**(k-1))

    # clamp seems to have backward gradient
    # https://github.com/pytorch/pytorch/blob/53fe804322640653d2dddaed394838b868ce9a26/torch/autograd/_functions/pointwise.py#L95
    x = torch.clamp(x, -limit, limit-1)

    # dequan
    x = (x + b) / delta

    return x


class SYQConv2d(nn.Module):

    def __init__(self, a_bit, w_bit, b_bit, momentum=0.9, fixed=False):
        super(SYQConv2d, self).__init__()

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.b_bit = b_bit
        self.momentum = momentum
        self.fixed = fixed
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))

    def fix_stat(self):

        self.fixed = True

    def un_fix_stat(self):

        self.fixed = False

    def convert_param(self, origin_conv):
        self.in_channels = origin_conv.in_channels
        self.out_channels = origin_conv.out_channels
        self.kernel_size = origin_conv.kernel_size
        self.stride = origin_conv.stride
        self.padding = origin_conv.padding
        self.dilation = origin_conv.dilation
        self.groups = origin_conv.groups
        self.weight = nn.Parameter(origin_conv.weight.data.clone())
        if origin_conv.bias is not None:
            self.bias = nn.Parameter(origin_conv.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):

        if not self.fixed:

            x_min = x.data.min()
            x_max = x.data.max()
            
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)
            

        # # x = SyQuan(x, self.a_bit, self.x_min, self.x_max, gradient=False)

        # qweight = SyQuan(self.weight, self.w_bit,
        #                  self.weight.min().data, self.weight.max().data)
        # if self.bias is not None:
        #     q_bias = SyQuan(self.bias, self.b_bit,
        #                     self.bias.min().data, self.bias.max().data)
        # else:
        #     q_bias = self.bias

        # return F.conv2d(x, qweight, q_bias,
        #                 self.stride, self.padding,
        #                 self.dilation, self.groups)
        
        w = self.weight
        x_transform = w.data.contiguous().view(self.out_channels, -1)
        
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        
        w = SyQuan(self.weight, self.w_bit,
                         w_min, w_max)
        
        
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)



class SYQLinear(nn.Module):
    def __init__(self, a_bit, w_bit, b_bit, momentum=0.9, fixed=False):
        super(SYQLinear, self).__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.b_bit = b_bit
        self.momentum = momentum
        self.fixed = fixed
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))

    def fix_stat(self):

        self.fixed = True

    def un_fix_stat(self):

        self.fixed = False

    def convert_param(self, origin_linear):
        self.in_features = origin_linear.in_features
        self.out_features = origin_linear.out_features
        self.weight = nn.Parameter(origin_linear.weight.data.clone())
        if origin_linear.bias is not None:
            self.bias = nn.Parameter(origin_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):

        if not self.fixed:

            # with torch.no_grad():
                # x_min = min(x.min()	, self.x_min) * self.momentum
                # x_max = max(x.max()	, self.x_max) * self.momentum

                # self.x_min.mul_(1-self.momentum).add_(x_min * self.momentum)
                # self.x_max.mul_(1-self.momentum).add_(x_max * self.momentum)
            x_min = x.data.min()
            x_max = x.data.max()
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)

        # x = SyQuan(x, self.a_bit, self.x_min, self.x_max, gradient=True)

        # qweight = SyQuan(self.weight, self.w_bit,
        #                  self.weight.min().data, self.weight.max().data)
        # if self.bias is not None:
        #     q_bias = SyQuan(self.bias, self.b_bit,
        #                     self.bias.min().data, self.bias.max().data)
        # else:
        #     q_bias = self.bias

        # return F.linear(x, qweight, q_bias)

        w = self.weight
        x_transform = w.data.detach()
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        
        w = SyQuan(self.weight, self.w_bit, w_min, w_max)
        
        
        return F.linear(x, weight=w, bias=self.bias)




class SYQActivation(nn.Module):

    def __init__(self, a_bit, momentum=0.9, fixed=False):
        super(SYQActivation, self).__init__()
        self.a_bit = a_bit
        self.momentum = momentum
        self.fixed = fixed
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))

    def fix_stat(self):

        self.fixed = True

    def un_fix_stat(self):

        self.fixed = False

    def forward(self, x):

        if not self.fixed:

            x_min = x.data.min()
            x_max = x.data.max()
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)

        # it seems that using gradient in activation will lead to nan loss of q model
        x = SyQuan(x, self.a_bit, self.x_min, self.x_max)

        return x
