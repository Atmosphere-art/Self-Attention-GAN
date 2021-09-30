# Self-Attention-GAN
Replicating 《Self-Attention Generative Adversarial Networks》 with PaddlePaddle  
使用PaddlePaddle复现《Self-Attention Generative Adversarial Networks》论文

## 一、简介
本项目基于paddlepaddle框架复现SAGAN，SAGAN是一种以标签为条件的图像生成网络。输入图像标签与随机产生的噪声，就可以生成对应标签的图像。  
### 论文
[1] Zhang, Han, et al. "Self-attention generative adversarial networks." International conference on machine learning. PMLR, 2019.

## 二、复现精度
模型在LSVRC2012(ImageNet)上进行测试，随机以1000类标签为条件生成了50000张图像进行评估，评估指标FID值和IS值如下所示：  
FID | 栏目2 

IS | 内容2 

### 预训练模型下载
下载地址：

## 三、数据集
LSVRC2012(ImageNet)数据集  
数据集大小：  
  训练集：1000个类别，一共1279591张图片  
  验证集：1000个类别，每个类别50张图片  
  测试集：10000张图片
  
## 四、环境依赖
python 3.7  
PaddlePaddle 2.1.1

## 五、快速开始
### 训练
多卡训练：python -m paddle.distributed.launch train.py --data_path '.../train'  
单卡训练：python train.py --data_path '.../train'  
说明：data_path为训练集路径

### 测试
python test.py --test_data_path '.../val'  
说明：test_data_path为验证集路径，用来随机读取标签
