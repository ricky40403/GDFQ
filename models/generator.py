
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from models.resblock import Block


import numpy as np
import torch
import torch.nn as nn





class TOYGenerator(nn.Module):
    """
    The classification generator.
    
    Args:
        n_class (int): the number of classes.
        in_channel (int): the input dim of the gaussian distribution.
        img_shape (tuple): the shape of output image.
                
    Return:
        generated image
    """
    
    def __init__(self, n_classes, in_channel, img_shape):
        
        super(TOYGenerator, self).__init__()
        
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.img_shape = img_shape
        self.layer_1 = nn.Sequential(nn.Linear(in_channel, 128))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )        
        
    
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.mul(self.label_emb(labels), noise)
        img = self.layer_1(gen_input)
        img = self.conv_blocks(img)        
        return img



class CLSGenerator(nn.Module):
    """
    The classification generator.
    
    Args:
        n_class (int): the number of classes.
        in_channel (int): the input dim of the gaussian distribution.
        img_shape (tuple): the shape of output image.
                
    Return:
        generated image
    """
    
    def __init__(self, n_classes, in_channel, img_size):
        
        super(CLSGenerator, self).__init__()
        
        self.label_emb = nn.Embedding(n_classes, in_channel)
        self.init_size = img_size // 4
        self.layer_1 = nn.Sequential(nn.Linear(in_channel, 128*(self.init_size**2)))        
        self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, stride=1, padding=1)
            )
        
        
    
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.layer_1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)        
        return img


class ResNetGenerator(nn.Module):
    """
    Ref from https://github.com/crcrpar/pytorch.sngan_projection/blob/master/models/generators/resnet.py
    using resnet generator to handle resnet
    """

    def __init__(self, num_features=64, dim_z=100, img_size=224,
                 activation=F.relu, num_classes=1000):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.init_size = img_size // 32
        self.activation = activation
        self.num_classes = num_classes        

        self.l1 = nn.Linear(dim_z, 16 * num_features * self.init_size ** 2)

        self.block2 = Block(num_features * 16, num_features * 16,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = Block(num_features * 16, num_features * 8,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = Block(num_features * 8, num_features * 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block5 = Block(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block6 = Block(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b7 = nn.BatchNorm2d(num_features)
        self.conv7 = nn.Conv2d(num_features, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.init_size, self.init_size)        
        for i in [2, 3, 4, 5, 6]:
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b7(h))
        return torch.tanh(self.conv7(h))