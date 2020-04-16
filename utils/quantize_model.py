
import copy

import torch
import torch.nn as nn

from utils.asy_quan import *
from models.conditional_batchnorm import CategoricalConditionalBatchNorm2d as CBN


def quantize_model(model, w_bit, b_bit, a_bit):
	
	if type(model) == nn.Conv2d:
		quan_layer = AsyQConv2d(a_bit, w_bit, b_bit)
		quan_layer.convert_param(model)
		return quan_layer

	elif type(model) == nn.Linear:
		quan_layer = AsyQLinear(a_bit, w_bit, b_bit)
		quan_layer.convert_param(model)
		return quan_layer

	elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
		return nn.Sequential(*[model, AsyQActivation(a_bit)])

	elif type(model) == nn.Sequential:

		modules = []
		for _, m in model.named_children():
			modules.append(quantize_model(m, w_bit, b_bit, a_bit))

		return nn.Sequential(*modules)


	else:
		q_model = copy.deepcopy(model)

		for m in dir(model):
			mod = getattr(model, m)
			if isinstance(mod, nn.Module) and "norm" not in m:
				setattr(q_model, m, quantize_model(mod, w_bit, b_bit, a_bit))

	return q_model



def freeze_bn(model):

	if isinstance(model, torch.nn.modules.BatchNorm2d):
		model.affine = False
		model.track_running_stats = False
		# model.eval()
	if isinstance(model, CBN):
		model.affine = False
		model.track_running_stats = False
		# model.eval()

	elif isinstance(model, nn.Sequential):
		mods = []
		for _, m in model.named_children():
			freeze_bn(m)
	else:
		for m in dir(model):
			mod = getattr(model, m)
			if isinstance(mod, nn.Module):
				setattr(model, m, freeze_bn(mod))

	return model


def freeze_act(model):
    if type(model) == AsyQActivation:
        model.fix_stat()
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            freeze_act(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_act(mod)
        return model


def un_freeze_act(model):    
    
    if type(model) == AsyQActivation:        
        model.un_fix_stat()        
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():            
            un_freeze_act(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                un_freeze_act(mod)
        return model