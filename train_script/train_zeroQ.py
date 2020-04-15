import torch
import torch.nn as nn


def train_zeroQ(fp_model, q_model, num_class=1000, batch_size = 64, img_size = 224,
			   warmup_epoch = 4, total_epoch = 400, iter_per_epoch = 200,
			   q_lr = 1e-4,
			   beta=0.1, gamma=1):
	pass
	