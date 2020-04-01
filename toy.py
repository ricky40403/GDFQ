import torch 
import torch.nn as nn
import numpy as np

import torchvision.models as models
from torch.autograd import Variable

from models.generator import TOYGenerator
import matplotlib.pyplot as plt
from torch.utils import data
import tqdm




# generate 500 data
noisex = np.random.uniform(-4, 4, 500)
noisey = np.random.uniform(-4, 4, 500)
GTdata = list(zip(noisex, noisey))
label = ((noisex * noisey) > 0) * 1
plt.figure()
plt.scatter(noisex, noisey, s=50, c=label, alpha=.5)
plt.savefig('real_data.png')


# train fp model first
FP_Model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(inplace=True),
    nn.Linear(16, 32),
    nn.BatchNorm1d(32, 0.8),
    nn.ReLU(inplace=True),
    nn.Linear(32, 16),
    nn.BatchNorm1d(16, 0.8),
    nn.ReLU(inplace=True),
    nn.Linear(16, 2),
)

FP_Model.cuda()
FP_Model.train()

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(FP_Model.parameters(), lr = 0.001, weight_decay=0.0001, momentum = 0.9)

# train FP model first
toy_dataset = data.TensorDataset(FloatTensor(GTdata), FloatTensor(label))
toy_dataloader = data.DataLoader(toy_dataset, batch_size = 10, shuffle = True)
# train 100 epoch
print("Train FP model")
for _ in tqdm.tqdm(range(50)):
    for (input_data, input_label) in toy_dataloader:
        
        input_data = input_data.cuda()
        input_label = input_label.cuda()
        
        pred = FP_Model(input_data)        
        loss = criterion(pred, input_label.long())        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
# generate noise data 
gaussian_noisex = np.random.normal(0, 1, 500)
gaussian_noisey = np.random.normal(0, 1, 500)
# normalize to -4, 4
gaussian_noisex = gaussian_noisex * 4 / (max(abs(gaussian_noisex)))
gaussian_noisey = gaussian_noisey * 4 / (max(abs(gaussian_noisey)))
gaussian_label =  np.random.uniform(0, 1, 500) > 0.5
plt.figure()
plt.scatter(gaussian_noisex, gaussian_noisey, s=50, c=label, alpha=.5)
plt.savefig('gaussian.png')
        
# Gassian_data = list(zip(noisex, noisey))
# toy_dataset = data.TensorDataset(FloatTensor(Gassian_data), FloatTensor(label))
# toy_dataloader = data.DataLoader(toy_dataset, batch_size = 10, shuffle = True)   
im_shape = (2)
toy_g = TOYGenerator(n_classes = 2, in_channel = 2, img_shape = im_shape)
toy_g.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_G = torch.optim.SGD(toy_g.parameters(), lr = 0.001, weight_decay=0.0001, momentum = 0.9)

print("Train generator")
# train 100 epoch
for _ in tqdm.tqdm(range(50)):
    
    optimizer_G.zero_grad()
    
    # generate noise data 
    # batch 10
    noisex = np.random.normal(0, 1, 10)
    noisey = np.random.normal(0, 1, 10)
    # normalize to -4, 4
    noisex = noisex * 4 / (max(abs(noisex)))
    noisey = noisey * 4 / (max(abs(noisey)))
    noise_data = FloatTensor(list(zip(noisex, noisey)))
    
    gt_label =  np.random.uniform(0, 1, 10) > 0.5
    gt_label = LongTensor(gt_label)
    
    # to cuda 
    noise_data = noise_data.cuda()
    gt_label = gt_label.cuda()      
    
    out_img = toy_g(noise_data, gt_label)
    
    label = FP_Model(out_img)
    
    g_loss = criterion(label, gt_label)    
    
    g_loss.backward()
    optimizer_G.step()
    
# test g

toy_g.eval()
test_data = FloatTensor(list(zip(gaussian_noisex, gaussian_noisey)))
test_label = LongTensor(gaussian_label)


fake_data_list = []
for idx, test_d in enumerate(test_data):
    test_d = test_d.unsqueeze(0)
    test_l = test_label[idx].unsqueeze(0)
    fake_data = toy_g(test_d, test_l)
    
    fake_data_list.append(fake_data.tolist()[0])
    
    
fake_x, fake_y = zip(*fake_data_list)
plt.figure()
plt.scatter(list(fake_x), list(fake_y), s=50, c=gaussian_label, alpha=.5)
plt.savefig('fake.png')



