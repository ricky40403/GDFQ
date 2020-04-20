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

## Experiment



<table>
<tr><th> model </th> <th> QuanType </th> <th> W/A bit </th> <th> top1 </th> <th> top5 </th></tr>
<tr><th rowspan="10"> resnet18 </th>
<tr><th colspan="2"> fp </th><th> 69.758 </th> <th> 89.078 </th></tr>
<tr><th rowspan="4"> zeroQ </th>
<th> 8/8 </th> <th> 69.230 </th> <th> 88.840 </th></tr>
<th> 4/8 </th> <th> 57.582 </th> <th> 81.182 </th></tr>
<th> 8/4 </th> <th> 1.130 </th> <th> 3.056 </th></tr>
<th> 4/4 </th> <th> 0.708 </th><th> 2.396  </th></tr></tr>
<tr><th rowspan="4"> GDFQ </th>
<th> 8/8 </th> <th>  </th> <th>  </th></tr>
<th> 4/8 </th> <th>  </th> <th>  </th></tr>
<th> 8/4 </th> <th>  </th> <th>  </th></tr>
<th> 4/4 </th> <th>  </th><th>   </th></tr></tr>
</table>

```
I also try to clone the [origin zeroQ repository](https://github.com/amirgholami/ZeroQ/blob/ba37f793dbcb9f966b58f6b8d1e9de3c34a11b8c/classification/utils/quantize_model.py#L36) and just set the all weight_bit to 4, the acc is about 10.  
And get about 24.16% by using pytorchcv. But 2.16 by using torchvision's model.
```


## Training

* The floating model using torchvision, so the architecture must fit the torchvisoin model name.  
You may reference https://pytorch.org/docs/stable/torchvision/models.html
* Batch size set the default batch size as 32

Default is 4 bit  
```
python train.py [imagenet path]

optional arguments:
-a , --arch     model architecture
-m , --method   zeroQ, GDFQ
--n_epochs      GDFQ's trainig epochs
--n_iter        training iteration per trainig epochs
--batch_size    batch size
--q_lr          learning rate of GDFQ's quantization model
--g_lr          learning rate of GDFQ's generator model
-qa             quantization activation bit
-qw             quantization weight bit
-qb             quantization bias bit

```

Ex: Training with resnet  
```
python train.py -a resnet18

```

Ex: Training with vgg16_bn with 8 bit activation, 8bit weight, 8 bit bias  
```
python train.py -a vgg16_bn -qa 8 -qw 8 -qb 8
```

### Issue
1. Question about the fixed batch norm of the Qmodel. (will not affect the training? or it needs to quantize the batch norm first?)  
2. The toy experiment can not generate the beautiful output, maybe something wrong. (Any advice or PR is welcome)  
3. The acc is wired when using 4 bit in zeroQ when using difference model source.
  
### Todo
- [x] add zeroQ traning.  
- [ ] Check the effect of the BNS and KL.   

### Note
The performace did reach the number in the paper.   
So it may have some bug for now.  
All the results are base on fake quantization, not the true low bit inference.  