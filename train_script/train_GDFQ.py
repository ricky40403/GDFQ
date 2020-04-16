
import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.generator import CLSGenerator
from train_script.utils import *
from utils.val import validation
from utils.quantize_model import freeze_bn

def adjust_learning_rate(optimizer, epoch, base_lr):    
    lr = base_lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def train_GDFQ(fp_model, q_model, val_dataloder, criterion, 
			   num_class=1000, batch_size = 256, img_size = 224,
			   warmup_epoch = 4, total_epoch = 400, iter_per_epoch = 200,
			   q_lr = 1e-6, g_lr = 1e-3,
			   beta=0.1, gamma=1):


	# handle with gpu and batchs
	n_gpu = torch.cuda.device_count()
	# because author may use multi-gpu training
	# and try to fit the origin fp model batch
	# larger iteration per epoch follow the rule https://arxiv.org/abs/1706.02677v1
	total_batch = n_gpu * batch_size
	default_iter = 200
	base_batch_size = 256
	# prevent out of bound
	base_batch_size = max(base_batch_size, total_batch)
	scale_factor = (base_batch_size//total_batch)
	train_iter = default_iter * scale_factor
	g_lr = g_lr/scale_factor
	q_lr = q_lr/scale_factor


	FloatTensor = torch.cuda.FloatTensor
	LongTensor = torch.cuda.LongTensor

	

	generator = CLSGenerator(num_class, 100, img_size)
	
	fp_model.cuda()
	# freeze fp model weight
	for param in fp_model.parameters():
		param.requires_grad = False

	generator.train()
	# generator.cuda()
	q_model.train()
	# q_model.cuda()

	fp_model = nn.DataParallel(fp_model).cuda()
	generator = nn.DataParallel(generator).cuda()
	q_model = nn.DataParallel(q_model).cuda()

	g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)
	q_optimizer = torch.optim.SGD(q_model.parameters(), lr=q_lr, momentum=0.9, weight_decay = 1e-4)	

	hooks, hook_handles, bn_stats = [], [], []
	# get number of BatchNorm layers in the model
	layers = sum([
	    1 if isinstance(layer, nn.BatchNorm1d) else 0
	    for layer in fp_model.modules()
	])

	eps = 0.8
	for n, m in fp_model.named_modules():
	    if isinstance(m, nn.Linear) and len(hook_handles) < layers:        
	        hook = output_hook()
	        hooks.append(hook)
	        hook_handles.append(m.register_forward_hook(hook.hook))
	    if isinstance(m, nn.BatchNorm1d):
	        # get the statistics in the BatchNorm layers        
	        bn_stats.append(
	            (m.running_mean.detach().clone().flatten().cuda(),
	                torch.sqrt(m.running_var + eps).detach().clone().flatten().cuda()))
	assert len(hooks) == len(bn_stats)

	# clear hook
	for hook in hooks:
	    hook.clear()

	criterion = nn.CrossEntropyLoss()

	for epoch in range(total_epoch):
		# both decay by 0.1 every 100 epoch
		adjust_learning_rate(g_optimizer, epoch, q_lr)
		adjust_learning_rate(q_optimizer, epoch, g_lr)

		pbar = tqdm.trange(train_iter)
		for _ in pbar:

			input_mean = torch.FloatTensor([0.0]).cuda()
			input_std = torch.FloatTensor([1.0]).cuda()

			fp_model.zero_grad()
			g_optimizer.zero_grad()

			train_gaussian_noise =  np.random.normal(0, 1, (batch_size, 100))
			train_gaussian_label =  np.random.randint(0, num_class, batch_size)
			input_data = Variable(FloatTensor(train_gaussian_noise)).cuda()
			input_label = Variable(LongTensor(train_gaussian_label)).cuda()

			fake_data = generator(input_data, input_label)
			fake_label = fp_model(fake_data)

			# BNS loss
			mean_loss = 0
			std_loss = 0
			# compute the loss according to the BatchNorm statistics and the statistics of intermediate output
			for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):            
			    tmp_output = hook.outputs
			    bn_mean, bn_std = bn_stat[0], bn_stat[1]
			    # get batch's norm 
			    tmp_mean = torch.mean(tmp_output , dim = 0)
			    tmp_std = torch.sqrt(torch.var(tmp_output, dim = 0) + eps)
			    mean_loss += own_loss(bn_mean, tmp_mean)
			    std_loss += own_loss(bn_std, tmp_std)
			tmp_mean = torch.mean(input_data, dim = -1)
			tmp_std = torch.sqrt(torch.var(input_data, dim = -1) + eps)

			mean_loss +=  own_loss(input_mean, tmp_mean)
			std_loss += own_loss(input_std, tmp_std)

			bns_loss = mean_loss + std_loss

			g_loss = criterion(fake_label, input_label)
			g_loss = g_loss + beta * bns_loss

			g_loss.backward()
			g_optimizer.step()

			# if warm, only update G
			if epoch < warmup_epoch:
				pbar.set_description("epoch: {}, g_lr: {} ==> warm up ==>  G_loss: {}".format(epoch, get_lr(g_optimizer), g_loss.item()))
				continue

			# train q model
			q_optimizer.zero_grad()
			fp_model.zero_grad()

			detach_fake_data = fake_data.detach()			
			q_result = q_model(detach_fake_data)
			q_loss = criterion(q_result, input_label)
			q_logit = F.log_softmax(q_model(detach_fake_data), dim = 1)
			with torch.no_grad():
				fp_logit = F.log_softmax(fp_model(detach_fake_data), dim = 1)
			kd_loss = F.kl_div(q_logit, fp_logit, reduction='batchmean')
			q_loss = q_loss + gamma * kd_loss
			

			q_loss.backward()
			q_optimizer.step()

			pbar.set_description("epoch: {}, G_lr:{},  G_loss: {}, Q_lr:{}, Q_loss: {}".format(epoch, 
															get_lr(g_optimizer) , g_loss.item(),
															get_lr(q_optimizer), q_loss.item()))
			
	
		if epoch % 20 == 0:
			q_top_1, q_top_5 = validation(val_dataloder, q_model, criterion)

			print(" ==>Current Top1: {}, Top5: {}\n".format(q_top_1, q_top_5))

		torch.save(q_model.state_dict(), "q_model.pkl")
		torch.save(generator.state_dict(), "generator.pkl")

	return q_model