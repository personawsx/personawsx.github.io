# 1、Last Time: Neural Networks
### （1）Linear score function（单层线性分类器）
$$f = Wx$$

### （2）2-layer Neural Network（两层全连接神经网络）
$$f = W_2 \max(0, W_1 x)$$

### （3）存在的问题
Problem: The spatial structure of images is destroyed!

### 问题说明
全连接层需要把二维图像展平成一维长向量，**丢失了图像原本的空间位置、局部相邻像素结构信息**，这是全连接处理图像的核心缺陷。

# 2、Next: Convolutional Neural Networks（卷积神经网络 CNN）
### 整体网络两大模块
### 模块1：卷积+池化层（图像特征提取主干）
> Convolution and pooling operators extract features while respecting 2D image structure
卷积、下采样算子提取图像特征，**完整保留图像二维空间结构**，解决全连接层展平丢失空间信息的缺陷。
包含两类核心操作：
1. Convolutions 卷积：用滑动卷积核提取局部纹理、边缘等空间特征，输出多通道特征图（Image Maps）
2. Subsampling 下采样（池化 Pooling）：压缩特征图尺寸，降低计算量，增强特征鲁棒性

### 模块2：全连接层（分类输出）
> Fully-Connected layers form an MLP at the end to predict scores
卷积提取完图像特征后，将特征图展平送入多层全连接网络（MLP），最终输出各类别预测得分。

### 网络流程结构
Input（原始图像）→ Convolutions → Subsampling（多层特征图 Image Maps）→ Fully Connected → Output（分类结果）
# 3、Present：Transformers have taken over
2017：Transformers for language tasks
2021：Transformers for vision tasks

# 4、Convolutional Networks

### 输入
RGB图像： $3 \times 32 \times 32$ （3通道，32高32宽）

### 卷积层参数
（1）卷积核 filters： $6 \times 3 \times 5 \times 5$
6个卷积核；3通道匹配输入；核大小5×5
（2）偏置 bias：6维向量，每个卷积核对应1个偏置

$$
z = \sum_{c,i,j} w_{c,i,j} \cdot x_{c,i,j} + b_k
$$

$b_k$：第k个卷积核对应的偏置

### 输出
6张激活图 activation maps，单张尺寸  $1 \times 28 \times 28$
堆叠后整体输出张量： $6 \times 28 \times 28$

### 尺寸计算（无padding、stride=1）

$$out = 32 - 5 + 1 = 28$$

一次移动一格

### 核心逻辑
每个卷积核滑动遍历图像，生成一张特征图；多个卷积核输出多通道特征图，完整保留图像空间二维结构。

<img width="2557" height="1325" alt="Image" src="https://github.com/user-attachments/assets/2287cda8-1791-44db-aed5-1000162eb2b2" />
由于卷积核的三维和数量不同，每次卷积后输出的三维有可能和初始输入不同，以实现新的三维激活。

### Problem：

卷积是线性运算，多个卷积层堆叠相当于一个卷积层。

### Solvtion：

在卷积层之间增加激活函数。

<img width="1406" height="688" alt="Image" src="https://github.com/user-attachments/assets/55776425-7aa2-43ff-b693-1081c7eb54be" />

### （1）如果想实现卷积前后的尺寸相同，可以通过填充0的方式来实现。

<img width="2441" height="1201" alt="Image" src="https://github.com/user-attachments/assets/19dbd017-1eab-4fe4-a22f-6fb32220d2e6" />

### （2）Receptive Fields（视野感受，更深的层能够学习更大的结构）

<img width="2480" height="1037" alt="Image" src="https://github.com/user-attachments/assets/590b757b-ddf4-4484-b83b-61e54ee9ed46" />

### （3）How to increase effective receptive fields？

#### a:
增大步幅，一次移动多格。

 $W$：输入特征图单边尺寸
 $K$：卷积核尺寸
 $P$：单边填充 Padding
 $S$：滑动步长 Stride

### 输出尺寸计算公式
$$
\text{Output} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$
### （4）filter中可学习的参量有哪些？
三维&bias

# 5、Pooling Layers: Another way to downsample

<img width="791" height="357" alt="Image" src="https://github.com/user-attachments/assets/0c0c1618-9dd4-4303-8006-0211ce1e4c26" />

### （1）张量变换示例
输入张量： $64 \times 224 \times 224$
经过池化层 pool 输出张量： $64 \times 112 \times 112$

### （2）核心规则
Given an input $C \times H \times W$, downsample each $1 \times H \times W$ plane
输入格式：通道数 C × 高度 H × 宽度 W
池化操作**单独对每一个通道的二维特征图做下采样**
通道数量 C 保持不变，仅压缩高、宽空间尺寸

### （3）直观尺寸变化
单通道特征图 $224 \times 224$ → 下采样后 $112 \times 112$，长宽各缩小一半

### （4）池化超参数 Hyperparameters
Kernel Size：池化窗口大小
Stride：窗口滑动步长
Pooling function：池化计算方式（Max / Average）

### 关键特点
池化不改变通道数量，仅压缩图像空间分辨率，属于无参数的下采样操作。
### （5）what is the method we use for downsampling？
（1）最大池化 Max Pooling（最常用）
窗口内取最大值作为输出，保留最强特征，会引入非线性。

<img width="2247" height="1063" alt="Image" src="https://github.com/user-attachments/assets/21e54385-cf29-41f8-93c3-a2b4fef88594" />

（2）平均池化 Avg Pooling
窗口内所有数字求平均值，平滑整体特征，不会引入非线性。

# 6、Convolution and Pooling: Translation Equivariance(平移等变性)

$$
\text{Conv}(\text{Translate}(X)) = \text{Translate}(\text{Conv}(X))
$$

含义：**先平移图像再卷积 = 先卷积再平移特征图**。
