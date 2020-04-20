# ref from official zeroQ
# https://github.com/amirgholami/ZeroQ/blob/ba37f793dbcb9f966b58f6b8d1e9de3c34a11b8c/classification/classification/distill_data.py
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
# *

import os
import torch
import torch.nn as nn
import copy
import tqdm
import torch.optim as optim
from train_script.utils import *
from utils.val import validation
from train_script.get_zeroQ_data import *
from utils.quantize_model import quantize_model, freeze_act, freeze_bn, un_freeze_act



def getDistilData(teacher_model,
                  dataset,
                  batch_size,
                  num_batch=1,
                  for_inception=False):
    """
	Generate distilled data according to the BatchNorm statistics in the pretrained single-precision model.
	Currently only support a single GPU.
	teacher_model: pretrained single-precision model
	dataset: the name of the dataset
	batch_size: the batch size of generated distilled data
	num_batch: the number of batch of generated distilled data
	for_inception: whether the data is for Inception because inception has input size 299 rather than 224
	"""

    # initialize distilled data with random noise according to the dataset
    dataloader = getRandomData(dataset=dataset,
                               batch_size=batch_size,
                               for_inception=for_inception)
    
    
    print('****** Data loaded ******')

    eps = 1e-6
    # initialize hooks and single-precision model
    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()
    ])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the convolutional layers to get the intermediate output after convolution and before BatchNorm.
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().cuda()))
    assert len(hooks) == len(bn_stats)

    print("Training ZeroQ Dataset with {} batch...".format(num_batch))    
    for i, gaussian_data in enumerate(dataloader):        
        
        if i == num_batch:
            break
        
        # initialize the criterion, optimizer, and scheduler
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=False,
                                                         patience=100)

        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in tqdm.tqdm(range(500)):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(gaussian_data)
            mean_loss = 0
            std_loss = 0

            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)
                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)
            tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 3,
                                                     -1),
                                  dim=2)
            tmp_std = torch.sqrt(
                torch.var(gaussian_data.view(gaussian_data.size(0), 3, -1),
                          dim=2) + eps)
            mean_loss += own_loss(input_mean, tmp_mean)
            std_loss += own_loss(input_std, tmp_std)
            total_loss = mean_loss + std_loss

            # update the distilled data
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

        refined_gaussian.append(gaussian_data.detach().clone())
        
        
    for handle in hook_handles:
        handle.remove()

    print("Training ZeroQ Dataset Done ...........")
    return refined_gaussian

# training


def train_zeroQ(fp_model, q_model, val_dataloder, batch_size=32, for_incep=False):
    
    dataloader = getDistilData(
        fp_model.cuda(),
        "imagenet",
        batch_size=batch_size,
        for_inception=for_incep)

    # Freeze BatchNorm statistics
    q_model.eval()
    q_model = q_model.cuda()
    q_model = un_freeze_act(q_model)

    # update activation
    with torch.no_grad():
        for _, inputs in enumerate(dataloader):
            if isinstance(inputs, list):
                inputs = inputs[0]
                inputs = inputs.cuda()
                _ = quantize_model(inputs)

    print("Zero Shot Quantization Finished .....")

    q_model = freeze_act(q_model)
    print("Eval after zeroQ update")
    q_top_1, q_top_5 = validation(val_dataloder, q_model)


    return q_model