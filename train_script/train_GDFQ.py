
import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.generator import ResNetGenerator
from train_script.utils import *
from utils.val import validation
from utils.quantize_model import *

def adjust_learning_rate(optimizer, epoch, base_lr):    
    lr = base_lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def train_GDFQ(fp_model, q_model, val_dataloder, 
			   num_class=1000, batch_size = 32, img_size = 224,
			   warmup_epoch = 4, total_epoch = 400, iter_per_epoch = 200,
			   q_lr = 1e-6, g_lr = 1e-3,
			   beta=0.1, gamma=1, for_incep=False):


	
	default_iter = 200	
	train_iter = default_iter	


	FloatTensor = torch.cuda.FloatTensor
	LongTensor = torch.cuda.LongTensor

	

	generator = ResNetGenerator(num_classes=num_class, dim_z=100, img_size=img_size)
	

	fp_model.cuda()
	# freeze fp model weight and bn
	for param in fp_model.parameters():
		param.requires_grad = False
	fp_model = freeze_bn(fp_model)	

	
	generator.train()
	q_model.train()
	q_model = freeze_bn(q_model)
	q_model = un_freeze_act(q_model)
	

	# fp_model = nn.DataParallel(fp_model).cuda()
	generator = nn.DataParallel(generator).cuda()
	q_model = nn.DataParallel(q_model).cuda()

	g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)
	q_optimizer = torch.optim.SGD(q_model.parameters(), lr=q_lr, momentum=0.9, weight_decay = 1e-4)	

	hooks, hook_handles, bn_stats = [], [], []
	# get number of BatchNorm layers in the model
	layers = sum([
	    1 if isinstance(layer, nn.BatchNorm2d) else 0
	    for layer in fp_model.modules()
	])

	eps = 0.8
	
	for n, m in fp_model.named_modules():
		
		# last layer (linear) does not follow with batch norm , so ignore linear linear
		if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:        
			hook = output_hook()
			hooks.append(hook)
			hook_handles.append(m.register_forward_hook(hook.hook))
		if isinstance(m, nn.BatchNorm2d):
			# get the statistics in the BatchNorm layers        
			bn_stats.append(
				(m.running_mean.detach().clone().flatten().cuda(),
					torch.sqrt(m.running_var + eps).detach().clone().flatten().cuda()))
	
	assert len(hooks) == len(bn_stats)

	

	criterion = nn.CrossEntropyLoss()

	for epoch in range(total_epoch):
		# both decay by 0.1 every 100 epoch
		adjust_learning_rate(g_optimizer, epoch, g_lr)
		adjust_learning_rate(q_optimizer, epoch, q_lr)

		pbar = tqdm.trange(train_iter)
		for _ in pbar:			

			input_mean = torch.zeros(1, 3).cuda()
			input_std = torch.ones(1, 3).cuda()

			fp_model.zero_grad()
			g_optimizer.zero_grad()

			train_gaussian_noise =  np.random.normal(0, 1, (batch_size, 100))
			train_gaussian_label =  np.random.randint(0, num_class, batch_size)
			input_data = Variable(FloatTensor(train_gaussian_noise)).cuda()
			input_label = Variable(LongTensor(train_gaussian_label)).cuda()

			fake_data = generator(input_data, input_label)	


			for hook in hooks:
				hook.clear()		
			fake_label = fp_model(fake_data)

			# BNS loss
			mean_loss = 0
			std_loss = 0
			# compute the loss according to the BatchNorm statistics and the statistics of intermediate output
			for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):            
				tmp_output = hook.outputs
				bn_mean, bn_std = bn_stat[0], bn_stat[1]
				# get batch's norm 
				tmp_mean = torch.mean(
					tmp_output.view(
						tmp_output.size(0),
						tmp_output.size(1),
						-1), dim=2)
				tmp_std = torch.sqrt(
								torch.var(
										tmp_output.view(tmp_output.size(0),
										tmp_output.size(1), -1),
										dim=2
									) + eps
							  )
				
				mean_loss += own_loss(bn_mean, tmp_mean)
				std_loss += own_loss(bn_std, tmp_std)

			
			tmp_mean = torch.mean(fake_data.view(fake_data.size(0), 3,-1), dim=2)
			tmp_std = torch.sqrt( torch.var(fake_data.view(fake_data.size(0), 3, -1), dim=2) + eps)

			mean_loss +=  own_loss(input_mean, tmp_mean)
			std_loss += own_loss(input_std, tmp_std)

			bns_loss = mean_loss + std_loss

			g_loss = criterion(fake_label, input_label)
			g_loss = g_loss + beta * bns_loss

			g_loss.backward()
			g_optimizer.step()		
					

			# train q model
			q_optimizer.zero_grad()
			fp_model.zero_grad()

			detach_fake_data = fake_data.detach()		
			# update activation
			q_result = q_model(detach_fake_data)

			
			if epoch >= warmup_epoch:

				q_loss = criterion(q_result, input_label)
				q_logit = F.log_softmax(q_model(detach_fake_data), dim = 1)
				with torch.no_grad():
					fp_logit = F.log_softmax(fp_model(detach_fake_data), dim = 1)
				kd_loss = F.kl_div(q_logit, fp_logit, reduction='batchmean')
				q_loss = q_loss + gamma * kd_loss	
			
				q_loss.backward()
				q_optimizer.step()

				pbar.set_description("epoch: {}, G_lr:{},  G_loss: {}, Q_lr:{}, Q_loss: {}".format(epoch+1, 
																get_lr(g_optimizer) , g_loss.item(),
																get_lr(q_optimizer), q_loss.item()))

			else:
				pbar.set_description("epoch: {}, g_lr: {} ==> warm up ==>  G_loss: {}".format(epoch+1, get_lr(g_optimizer), g_loss.item()))
		
		

		if (epoch+1) < warmup_epoch:
			pass
		elif (epoch+1) == warmup_epoch:			
			print("Free activaiton after warm up")
			q_model = freeze_act(q_model)
			print("Eval after warmup")
			q_top_1, q_top_5 = validation(val_dataloder, q_model, criterion)
		else:
			if (epoch+1) % 10 == 0:
				q_top_1, q_top_5 = validation(val_dataloder, q_model, criterion)
			
		torch.save(q_model.state_dict(), "q_model.pkl")
		torch.save(generator.state_dict(), "generator.pkl")



	for handle in hook_handles:
		handle.remove()
		

	return q_model