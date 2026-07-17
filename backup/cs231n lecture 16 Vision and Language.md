 
<img width="961" height="432" alt="Image" src="https://github.com/user-attachments/assets/252e661d-2d1c-4436-9b04-fa7a39cc8ac2" />

<img width="868" height="421" alt="Image" src="https://github.com/user-attachments/assets/5be79945-996b-4037-ac5c-63c816c64017" />

<img width="917" height="410" alt="Image" src="https://github.com/user-attachments/assets/32579b32-3922-447c-8cf0-9f4e0b30ff13" />

<img width="983" height="327" alt="Image" src="https://github.com/user-attachments/assets/8e34bdaa-8615-4252-9758-bfcd02184ffe" />

#### How do identify a model as a Foundation（基础模型）?
Always see with foundation models:
general /robust to many different tasks

#### Often see with foundation models:
Large # params
Large amount of data
Self-supervised pre-training objective

自监督学习的目标：
将带有同类元素的照片和文字归为一类
<img width="950" height="430" alt="Image" src="https://github.com/user-attachments/assets/4b496669-809f-430c-a39c-5153673d7898" />

# 1、SimCLR VS CLIP
<img width="897" height="500" alt="Image" src="https://github.com/user-attachments/assets/37ef3912-7ee4-4cae-834f-e3c9e6ecdd1d" />

SimCLR：只能实现图像的对比学习
CLIP：跨模态图文对比学习


<img width="975" height="435" alt="Image" src="https://github.com/user-attachments/assets/42f2aef0-b14e-40de-a847-80297feda334" />

CLIP：先用海量图文对做无监督对比预训练，得到通用图像编码器，再拿这个编码器配合简单线性层，直接完成分类、检测、分割等各类视觉任务
 
Problem：想要做图像分类 / 检测 / 分割，必须提供该任务的标注数据，训练这个新增线性层，即不支持 Zero-Shot（零样本）


### LLM的不同之处：支持 Zero-Shot（零样本）

<img width="987" height="495" alt="Image" src="https://github.com/user-attachments/assets/724a1c1f-c91c-4ae0-8893-f6e7c8d6d008" />

<img width="962" height="462" alt="Image" src="https://github.com/user-attachments/assets/769b2c9a-dc6d-4f39-98e4-33a66534fb71" />

Question：视觉语言模型（CLIP）怎么零样本使用？

<img width="633" height="323" alt="Image" src="https://github.com/user-attachments/assets/187cfadd-37ac-4be1-b63e-4f35cc959136" />

### Solution：

待识别图片送入 Image Encoder，得到图像向量 image vector；
类别文字（比如 "Dog"等）送入 Text Encoder，得到文本向量 text vector；
计算两个向量的相似度匹配分数（图里 0.27），分数越高代表图片和这个文字描述越匹配（1-NN 近邻思想）。

### improvement：

#### （1）使用完整描述短语，提升精度；
<img width="647" height="307" alt="Image" src="https://github.com/user-attachments/assets/1c10cb44-c641-4991-8e47-58904b309a00" />

#### （2）多短语消除偏见
<img width="623" height="307" alt="Image" src="https://github.com/user-attachments/assets/bb80f15c-e6e7-4219-9a3a-af344c79f82e" />

