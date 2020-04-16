# GDFQ
unofficial implementation of Generative Low-bitwidth Data Free Quantization  
It is for personal study, any advice is welcome.  


****  
zeroQ: <https://arxiv.org/pdf/2001.00281.pdf>  
The Origin Paper : <https://arxiv.org/pdf/2003.03603.pdf>  
****


## Toy Result

<img src=toy.png width=100%>  

It seems that the Generator can generate data with classification boundary,
and zeroQ will prefer to generate data that will not pay attention to the data distribution.  

But I can not reproduce the beautiful generated data in the paper. :sweat_smile::sweat_smile::sweat_smile:  


## Training

* The floating model using torchvision, so the architecture must fit the torchvisoin model name.  
* Batch size set the default batch size as 256, and it will follow the related rules of the learning rate, iteration, batch size.  
Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour: <https://arxiv.org/abs/1706.02677v1>   

Default is 4 bit  
Ex: training with resnet  
```

python train.py -a resnet18 --batch_size 64
```

Training with vgg16_bn with 8 bit activation, 8bit weight, 8 bit bias  
```
python train.py -a vgg16_bn -qa 8 -qw 8 -qb 8
```

### Issue
1. Question about the fixed batch norm of the Qmodel. (will not affect the training? or it needs to quantize the batch norm first?)  
2. The toy experiment can not generate the beautiful output, maybe something wrong. (Any advice or PR is welcome)  
  
### Todo
- [ ] add zeroQ traning.  
- [ ] Check the effect of the BNS and kl(if have time).   

### Note
This had not tested the performance yet.   
So it may have some bug for now.  
All the results are base on fake quantization, not the true low bit inference.  