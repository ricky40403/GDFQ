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

def AsyQuan(x, k, x_min, x_max):
	
	
	# prevent 0 in denominator
	scale = (2**(k-1)) / torch.clamp((x_max - x_min), min=1e-10)
	
	zero_point = scale * x_min
	
	zero_point += (2**(k-1))

	zero_point = zero_point.round()
	
	# quan
	x = scale * x - zero_point
	
	# x = RoundWithGradient.apply(x)
	x = x.round()
	
	limit = (2**(k-1))
	# x = torch.where(x < (-limit), torch.tensor([-limit]).float().cuda(), x)
	# x = torch.where(x > (limit-1), torch.tensor([limit-1]).float().cuda(), x)	
	
	# clamp seems to have backward gradient
	# https://github.com/pytorch/pytorch/blob/53fe804322640653d2dddaed394838b868ce9a26/torch/autograd/_functions/pointwise.py#L95
	x = torch.clamp(x, -limit, limit-1)
	# dequan
	x = (x + zero_point) / scale
	# print("x in asy 4: {}".format(x[0][0][0][0]))

	return x




class AsyQConv2d(nn.Module):

	def __init__(self, a_bit, w_bit, b_bit):
		super(AsyQConv2d, self).__init__()

		self.w_bit = w_bit
		self.a_bit = a_bit
		self.b_bit = b_bit


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

		q_a = AsyQuan(x, self.a_bit, torch.min(x), torch.max(x))
		
		qweight = AsyQuan(self.weight, self.w_bit, torch.min(self.weight), torch.max(self.weight))
		if self.bias is not None:
			q_bias = AsyQuan(self.bias, self.b_bit, torch.min(self.bias), torch.max(self.bias))
		else:
			q_bias = self.bias

		
		return  F.conv2d(q_a, qweight, q_bias,
						self.stride, self.padding,
                        self.dilation, self.groups)


class AsyQLinear(nn.Module):
	def __init__(self, a_bit, w_bit, b_bit):
		super(AsyQLinear, self).__init__()
		self.w_bit = w_bit
		self.a_bit = a_bit
		self.b_bit = b_bit


	def convert_param(self, origin_linear):
		self.in_features = origin_linear.in_features
		self.out_features = origin_linear.out_features
		self.weight = nn.Parameter(origin_linear.weight.data.clone())   
		if origin_linear.bias is not None:
			self.bias = nn.Parameter(origin_linear.bias.data.clone())
		else:
			self.bias = None


	def forward(self, x):
		
		q_a = AsyQuan(x, self.a_bit, torch.min(x), torch.max(x))
		qweight = AsyQuan(self.weight, self.w_bit, torch.min(self.weight), torch.max(self.weight))
		if self.bias is not None:
			q_bias = AsyQuan(self.bias, self.b_bit, torch.min(self.bias), torch.max(self.bias))

		else:
			q_bias = self.bias

		
		return F.linear(q_a, qweight, q_bias)


class AsyQActivation(nn.Module):

	def __init__(self, a_bit, momentum = 0.9, fixed = False):
		super(AsyQActivation, self).__init__()
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

			x_min = x.min()	
			x_max = x.max()			

			self.x_min = self.x_min * (1-self.momentum) + x_min * self.momentum
			self.x_max = self.x_max * (1-self.momentum) + x_max * self.momentum		

		
		x = AsyQuan(x, self.a_bit, self.x_min, self.x_max)	


		return x