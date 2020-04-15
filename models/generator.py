
import numpy as np
import torch
import torch.nn as nn





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
