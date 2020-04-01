
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
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(in_channel + n_classes, 16, normalize=False),
            *block(16, 32),
            *block(32, 32),
            *block(32, 16),
            nn.Linear(16, int(np.prod(img_shape))),
            # nn.Tanh()
        )
        
    
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)        
        img = self.model(gen_input)
        img = img.view(img.size(0), 2)
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
    
    def __init__(self, n_classes, in_channel, img_shape):
        
        super(CLSGenerator, self).__init__()
        
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(in_channel + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        
    
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)        
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img
