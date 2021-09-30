# Self-Attention-GAN
Replicating 《Self-Attention Generative Adversarial Networks》 with PaddlePaddle  
使用PaddlePaddle复现《Self-Attention Generative Adversarial Networks》论文

## 一. 简介
本项目基于paddlepaddle框架复现SAGAN，SAGAN是一种以标签为条件的图像生成网络。输入图像标签与随机产生的噪声，就可以生成对应标签的图像。  
### 论文
[1] Zhang, Han, et al. "Self-attention generative adversarial networks." International conference on machine learning. PMLR, 2019.

## 二.复现精度
模型在LSVRC2012(ImageNet)上进行测试，随机以1000类标签为条件生成了50000张图像进行评估，评估指标FID值和IS值如下所示：  
FID | 栏目2 

IS | 内容2 
