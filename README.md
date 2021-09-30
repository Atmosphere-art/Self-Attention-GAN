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

### 测试
python test.py --test_data_path '.../val' --pretrained_model '.../sagan_paddle_pretrained.pdparams'

## 六、代码结构与详细说明
### 代码结构
├─dataset.py                                            # 读取数据集  
├─parameters.py                                         # 参数设置  
├─sagan_models.py                                       # 模型  
├─test.py                                               # 测试  
├─tester.py                                             # 测试  
├─train.py                                              # 训练  
├─trainer.py                                            # 训练  
├─utils.py                                              # 图像保存,dataloader创建等方法  
├─sagan_models                                          # 存储日志等内容的文件夹  
|  samplers                                             # 训练时生成的图片结果  
|  weights                                              # 训练时保存的checkpoint  
|  log.txt                                              # 训练日志  

### 参数说明
data_path：训练集路径  
test_data_path：验证集路径，用来随机读取标签  
pretrained_model：测试时所加载的模型路径  

## 七、模型信息
信息 | 说明 

发布者 | 王仁君

时间 | 2021.09

框架版本 | PaddlePaddle 2.1.1

应用场景 | 图像生成

支持硬件 | CPU、GPU
