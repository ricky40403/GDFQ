import torch 
import torch.nn as nn
import numpy as np

import torchvision.models as models
from torch.autograd import Variable

from models.generator import TOYGenerator
import matplotlib.pyplot as plt
from torch.utils import data
import tqdm
from scipy.stats import truncnorm
import torch.optim as optim

import random


def own_loss(A, B):
    """
	L-2 loss between A and B normalized by length.
    Shape of A should be (features_num, ), shape of B should be (batch_size, features_num)
	"""
    return (A - B).norm()**2 / B.size(0)


class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None


test_data_size = 500
test_batch_size = 10
test_epoch = 500

# generate 500 data
noisex = np.random.uniform(-4, 4, test_data_size)
noisey = np.random.uniform(-4, 4, test_data_size)
GTdata = list(zip(noisex, noisey))
label = ((noisex * noisey) > 0) * 1
plt.figure()
# plt.subplot(1, 4, 1)
plt.scatter(noisex, noisey, s=50, c=label, alpha=.5)
plt.savefig('real_data.png')
# exit()


# train fp model first
FP_Model = nn.Sequential(
    nn.Linear(2, 128),
    nn.BatchNorm1d(128, 0.8),
    nn.ReLU(inplace=True),
    nn.Linear(128, 256),
    nn.BatchNorm1d(256, 0.8),
    nn.ReLU(inplace=True),
    nn.Linear(256, 512),
    nn.BatchNorm1d(512, 0.8),
    nn.ReLU(inplace=True),
    nn.Linear(512, 2),
)



FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(FP_Model.parameters(), lr = 0.01, weight_decay=0.0001, momentum = 0.9)

# train FP model first
toy_dataset = data.TensorDataset(FloatTensor(GTdata), FloatTensor(label))
toy_dataloader = data.DataLoader(toy_dataset, batch_size = 10, shuffle = True)

FP_Model.cuda()
FP_Model.train()
# train 100 epoch
print("Train FP model")
for _ in tqdm.tqdm(range(50), position = 0, leave=True):
    for (input_data, input_label) in toy_dataloader:
        
        input_data = input_data.cuda()
        input_label = input_label.cuda()
        
        pred = FP_Model(input_data)        
        loss = criterion(pred, input_label.long())      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    
# generate noise data 
# gaussian_noisex = truncnorm(a = -4, b = 4).rvs(size=500)
# gaussian_noisey = truncnorm(a = -4, b = 4).rvs(size=500)
gaussian_noise =  np.random.normal(0, 1, (test_data_size, 2))
# gaussian_label =  np.random.uniform(0, 1, test_data_size) > 0.5
gaussian_label = np.random.randint(0, 1, size = test_data_size)
plt.figure()
# plt.subplot(1, 4, 2)
plt.scatter(gaussian_noise[:, 0], gaussian_noise[:, 1], s=50, c=label, alpha=.5)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.savefig('gaussian.png')




gaussian_input_data = list(gaussian_noise)
random.shuffle(gaussian_input_data)

toy_dataset = data.TensorDataset(FloatTensor(gaussian_input_data), LongTensor(gaussian_label))
toy_dataloader = data.DataLoader(toy_dataset, batch_size = test_batch_size, shuffle = True)


print("ZeroQ training...")
# zero Q does not need input_label
# reference from https://github.com/amirgholami/ZeroQ/blob/f91092e0dfee676f4f667e04b2228543b27f9ce1/distill_data.py#L53
FP_Model = FP_Model.eval()

hooks, hook_handles, bn_stats = [], [], []
# get number of BatchNorm layers in the model
layers = sum([
    1 if isinstance(layer, nn.BatchNorm1d) else 0
    for layer in FP_Model.modules()
])

eps = 0.8
for n, m in FP_Model.named_modules():
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


zeroQ_gaussian_data = []
zeroQ_gaussian_label = []

for idx, (input_data, input_label) in enumerate(toy_dataloader):    

    input_data = input_data.cuda()
    input_data.requires_grad = True    
    FP_Model.zero_grad()
    crit = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam([input_data], lr = 0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        min_lr=1e-4,
                                                        verbose=False,
                                                        patience=100)
    
    input_mean = torch.FloatTensor([0.0]).cuda()
    input_std = torch.FloatTensor([1.0]).cuda()

    for _ in tqdm.tqdm(range(test_epoch)):

        FP_Model.zero_grad()
        optimizer.zero_grad()
        # clear hook
        for hook in hooks:            
            hook.clear()
        
        output = FP_Model(input_data)
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

        mean_loss += own_loss(input_mean, tmp_mean)
        std_loss += own_loss(input_std, tmp_std)
        total_loss = mean_loss + std_loss        
        
        # update the distilled data
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        

    zeroQ_gaussian_data.extend(input_data.detach().clone().tolist())
    zeroQ_gaussian_label.extend(input_label.detach().clone().tolist())


for handle in hook_handles:
    handle.remove()

# gather to x, y
zeroQ_x, zeroQ_y = list(zip(*zeroQ_gaussian_data))
plt.figure()
plt.scatter(list(zeroQ_x), list(zeroQ_y), s = 50, c = zeroQ_gaussian_label, alpha = .5)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.savefig('zeroQ.png')
# exit()

print("Train Genrative data")

im_shape = (2)
toy_g = TOYGenerator(n_classes = 2, in_channel = 2, img_shape = im_shape)
toy_g.train()
toy_g = toy_g.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(toy_g.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    min_lr=1e-4,
                                                    verbose=False,
                                                    patience=100)



print("Generator warm up")


FP_Model = FP_Model.eval()
hooks, hook_handles, bn_stats = [], [], []
# get number of BatchNorm layers in the model
layers = sum([
    1 if isinstance(layer, nn.BatchNorm1d) else 0
    for layer in FP_Model.modules()
])
eps = 0.8
for n, m in FP_Model.named_modules():
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

FP_Model.zero_grad()
# set same iteration as zeroQ
for _ in tqdm.tqdm(range(test_epoch * (test_data_size//test_batch_size))):

    optimizer.zero_grad()

    gaussian_noise =  np.random.normal(0, 1, (test_batch_size, 2))    
    # randint is discrete uniform    
    gaussian_label = np.random.randint(0, 1, size = test_batch_size)
    

    input_mean = torch.FloatTensor([0.0]).cuda()
    input_std = torch.FloatTensor([1.0]).cuda()

    input_data = Variable(FloatTensor(gaussian_noise))        
    input_label = Variable(LongTensor(gaussian_label))

    input_data = input_data.cuda()
    input_label = input_label.cuda()

    fake_data = toy_g(input_data, input_label)

    fake_label = FP_Model(fake_data)

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

    g_loss = criterion(fake_label, input_label)

    total_loss = g_loss + 1 * (mean_loss + std_loss)

    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss.item())


# # test toy_g
# # gaussian_noisex = truncnorm(a = -4, b = 4).rvs(size=500)
# # gaussian_noisey = truncnorm(a = -4, b = 4).rvs(size=500)
# gaussian_noisex =  np.random.normal(0, 1, 500)
# gaussian_noisey =  np.random.normal(0, 1, 500)
# gaussian_input_data = list(zip(gaussian_noisex, gaussian_noisey))
# gaussian_label =  np.random.uniform(0, 1, 500) > 0.5

gaussian_noise =  np.random.normal(0, 1, (test_data_size, 2))    
gaussian_label = np.random.randint(0, 1, size = test_data_size)  
toy_g.eval()
test_inputs = FloatTensor(gaussian_noise)
test_labels = FloatTensor(gaussian_label)

fake_data_list = []
for idx, test_d in enumerate(test_inputs):

    test_input = test_d.unsqueeze(0).cuda()
    test_label = test_labels[idx].unsqueeze(0).cuda()
    fake_data = toy_g(test_input, test_label.long())
    # print(fake_data.tolist()[0])
    fake_data_list.append(fake_data.tolist()[0])


fake_x, fake_y = zip(*fake_data_list)
plt.figure()
plt.scatter(list(fake_x), list(fake_y), s = 50, c = gaussian_label, alpha = .5)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.savefig('fake.png')

