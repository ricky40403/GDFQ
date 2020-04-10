# GDFQ
unofficial implementation of Generative Low-bitwidth Data Free Quantization


****  
zeroQ: <https://arxiv.org/pdf/2001.00281.pdf>  
The Origin Paper : <https://arxiv.org/pdf/2003.03603.pdf>  
****


---
## Toy Result

<img src=toy.png width=100%>  
Real Data ==> train FP model.  
Gaussian Data ==> Generated gaussian to train zeroQ/generator and test output.  
ZeroQ ==> Data distribution generated from zeroQ method.  
Fake data ==> Data distribution generated from the generator.  


It seems that the Generator can generate data with classification boundary,
and zeroQ will prefer to generate data that will not pay attention to the data distribution.

But I can not reproduce the beautiful generated data in the paper. :sweat_smile::sweat_smile::sweat_smile:


