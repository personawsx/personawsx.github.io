# Lecture Overview – Two Broad Sets of Topics
# 课程总览：两大核心主题模块

# 一、How to build CNNs? 如何搭建卷积神经网络
（1）Layers in CNNs：卷积网络各类层（卷积层、池化层、全连接层等）
（2）Activation Functions：激活函数（ReLU等非线性变换）
（3）CNN Architectures：经典CNN网络结构（LeNet、AlexNet、VGG等）
（4） Weight Initialization：权重初始化方法

# 二、How to train CNNs? 如何训练卷积神经网络
（1）Data Preprocessing：数据预处理
（2）Data augmentation：数据增强
（3）Transfer Learning：迁移学习
（4）Hyperparameter Selection：超参数选择

# 1、Normalization

<img width="798" height="245" alt="Image" src="https://github.com/user-attachments/assets/d775b293-8d2c-4df4-a49c-d5d72116c7b2" />

#### 维度字母释义
$N$：Batch 批量样本数
$C$：Channel 通道数
$H$：Height 特征图高度
$W$：Width 特征图宽度

#### 四种归一化统计区间
（1）BatchNorm：同一通道，跨全部样本 N,H,W
（2）LayerNorm：单样本，跨全部通道 C,H,W
（3）InstanceNorm：单样本单通道，仅 H,W
（4）GroupNorm：单样本分通道组，每组内 H,W

# 2、Regularization：Dropout
### （1）训练阶段原理
In each forward pass, randomly set some neurons to zero
每一轮前向传播时，随机将一部分神经元输出置0，使其失效。
- `dropping probability`：失活概率，属于超参数，常用值 0.5（一半神经元随机关闭）。

#### 图示对比

<img width="561" height="247" alt="Image" src="https://github.com/user-attachments/assets/119cc3fc-ace0-4d86-9507-7cfd4e380892" />

左图：无Dropout，一层所有神经元全部参与计算；
右图：开启Dropout，带叉号神经元被随机屏蔽，输出置0，不参与本轮前向、反向传播。

### （2）Dropout 为什么能防止过拟合
#### a：Forces the network to have a redundant representation.
迫使网络学习冗余、多元的特征表达，不依赖少数几个神经元；
#### b：Prevents co-adaptation of features.
阻止特征之间互相依赖（协同适应）。

#### 通俗举例
识别猫咪的神经元：`有耳朵、有尾巴、毛茸茸`
无Dropout：模型可能只记住“毛茸茸”这一个特征，其他特征不学习；
有Dropout：每轮随机屏蔽部分神经元，模型无法依赖单一特征，必须学会全部特征组合判断，泛化能力更强。

#### 本质效果
每次前向传播等价于随机生成一个小型子网络，多次迭代相当于**多个子网络集成学习**，降低单条特征过拟合风险。

### （3）测试/推理阶段逻辑
#### a：测试时**关闭Dropout**，所有神经元全部激活，不再随机屏蔽；
#### b：必须对激活值做缩放，保证：
$$\text{测试时神经元输出} = \text{训练时神经元期望输出}$$

```python
def predict(X):
    # 推理阶段，所有神经元激活，乘以保留概率p缩放
    H1 = np.maximum(0, np.dot(W1, X) + b1) * p
    H2 = np.maximum(0, np.dot(W2, H1) + b2) * p
    out = np.dot(W3, H2) + b3
```
# 3、Activation Function

### （1）Sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

### 缺点
a： 多层堆叠时梯度会持续变小；
b： $x\gg0$ 和  $x\ll0$ 两端饱和区，导数趋近于0。

### （2）ReLU（CNN标配）

$$f(x) = \max(0, x)$$

### 优点
正数区间无饱和，梯度稳定，不会梯度消失。
### 缺点
 **Dead ReLU（死亡神经元）**：输入 $x<0$ 时输出恒为0，梯度直接为0，该神经元永久不再更新，彻底失效。

### （3）GELU 高斯误差线性单元（Transformer）

$$f(x) = x\cdot \Phi(x)$$
$\Phi(x)$ 为标准高斯分布累积分布函数
### 优点
a：在0附近曲线平滑，过渡自然；
b：负数区域仍保留微弱梯度，解决ReLU死亡神经元问题。

### 缺点
a：包含高斯积分运算，计算开销远大于ReLU；
b：输入负无穷时梯度依旧会趋近于0，极端值仍存在梯度衰减。

在CNN中，一般将激活函数放在线性操作之后。

# 4、Case Study

### （1）VGGNet
<img width="2341" height="1187" alt="Image" src="https://github.com/user-attachments/assets/1f51281f-dd1e-413a-87cd-ee0aa050b6df" />

3个3×3（步幅为1）卷积核堆叠，与7×7的卷积核具有等效感受视野，且参数更少，故这样选取（这种解释很直觉，但无法知道作者的想法）。

### （2）ResNet

### ResNet 残差网络核心原理解析
#### a：背景问题：
网络层数加深后，训练、测试精度反而下降。

#### b：ResNet 核心思路
Use network layers to fit a residual mapping instead of directly trying to fit a desired underlying mapping
不让卷积层直接学习目标映射 $H(x)$，而是让卷积层学习**残差映射** $F(x) = H(x)-x$
最终输出公式： $$H(x) = F(x) + x$$

<img width="807" height="312" alt="Image" src="https://github.com/user-attachments/assets/5b0b22c0-186e-4509-ba94-e68023cd0bd0" />

前面网络已经提取完有效特征，当前这两层不需要新增特征，最优结果就是输出 = 输入，也就是最终要让 $H(x)=x$。

#### 左侧：普通 Plain 网络块
输入 $x$ → 两层卷积+ReLU → 直接输出 $H(x)$
网络需要从零学习完整变换 $H(x)$，层数很深时很难拟合恒等变换 $H(x)=x$。

#### 右侧：ResNet 残差块
同样目标 $H(x)=x$，代入差值公式： $F(x) = H(x)-x$
现在卷积层只需要学到输出全 0，整个残差块最终就可以输出 $H(x)$。

# 5、Weight Initialization

```python
dims = [4096] * 7
hs = []
x = np.random.randn(16, dims[0])
# Forward pass with ReLU activation
for Din, Dout in zip(dims[:-1], dims[1:]):
    W = 0.01 * np.random.randn(Din, Dout) # Small weight init
    x = np.maximum(0, x.dot(W)) # ReLU activation
    hs.append(x)
```
权重标准差取0.01过小，会导致梯度消失；取0.05时过大，会导致梯度爆炸。
### Solution:
```python
import numpy as np

dims = [4096] * 7
hs = []
x = np.random.randn(16, dims[0])

for Din, Dout in zip(dims[:-1], dims[1:]):
    # Kaiming He初始化，ReLU修正系数 sqrt(2 / Din)
    W = np.random.randn(Din, Dout) * np.sqrt(2 / Din)
    x = np.maximum(0, x.dot(W))
    hs.append(x)
```
即： $$std(W) = \sqrt{\frac{2}{D_{in}}}$$
其中 $D_{in}$ 为输入维度，在这里为4096。

# 6、Data Preprocessing
在模型处理前，减均值，除以平方差。

# 7、Data augmentation
### （1）原理
### 训练阶段 Training：引入随机噪声 z
> Training: Add some kind of randomness

公式表达：

$$
y = f_W(x, z)
$$

$x$：模型输入样本
$W$：网络权重参数
$z$：训练时额外引入的随机变量（随机源）
$f_W(\cdot)$：带随机扰动的网络前向函数

### 测试阶段 Testing：消除随机，取期望均值
> Testing: Average out randomness (sometimes approximate)

公式表达：

$$
y = f(x) = \mathbb{E}_z\left[f_W(x,z)\right] = \int p(z) f_W(x,z) dz
$$

$\mathbb{E}_z[\cdot]$：对随机变量 $z$ 求数学期望
$p(z)$：随机变量 $z$ 的概率分布
积分含义：遍历所有可能的随机扰动，对输出结果平均
补充说明：`So sometimes this is approximate`
精确积分计算期望成本极高，工程上会用**近似缩放**替代精确积分，降低计算开销。

### （2）Dropout 正则化
### 训练时：
> Training: Randomly drop activations

前向传播时，随机把一部分神经元激活值置0，随机掩码就是公式里的 $z$。

### 测试时
> Testing: Use all activations and average values

测试阶段不丢弃任何神经元。

### （3）其他方法
#### a：水平翻转（不包括文本）。
#### b：随机剪裁/缩放。
#### c：颜色抖动（randomize contrast and brightness，随机对比度和亮度）。 
#### d：裁剪或遮盖掉图像的一部分。

# 8、Transfer Learning（小数据集下CNN训练方案：迁移学习 ）
### （1）如果数据集规模很小，我们还能正常训练卷积神经网络吗？
### 解决方案：
**迁移学习（Transfer Learning）**：利用大型公开数据集（ImageNet）预训练好的CNN通用视觉特征，复用至小数据集任务，避免从零训练导致严重过拟合。

### （2）CNN迁移学习两种流程

<img width="1416" height="666" alt="Image" src="https://github.com/user-attachments/assets/c93ca6ae-13c5-4092-92a7-84258afe093f" />

#### 前置基础：已有预训练模型。

#### 方案1：极小数据集场景 — 冻结卷积层，仅训练顶层分类头
适用：样本数量极少，直接微调全部层极易过拟合

### 方案2：较大规模数据集场景 — 全网络微调（Fine-tune）
适用：自有数据量充足，和预训练数据集分布接近

<img width="1622" height="1113" alt="Image" src="https://github.com/user-attachments/assets/687ed045-7a48-4bf8-a114-9220a0a8cd72" />

### Fine-tune 微调
加载 ImageNet 预训练权重作为网络初始参数。


### Train from scratch 从零训练
不加载任何预训练权重，所有卷积、全连接权重随机初始化。

# 9、Hyperparameters Selection
### Step 1: Check initial loss 检查初始损失
模型刚初始化、未训练时，验证理论基线损失是否匹配。
分类任务中，随机初始化的模型预测概率均匀，初始交叉熵损失应等于 $-\log(1/C)$（$C$ 为类别数）；若初始损失远大于理论值，说明损失函数、标签、初始化存在错误。

### Step 2: Overfit a small sample 用少量样本过拟合测试
取极小一部分训练集（几十张图），关闭所有正则化，完整训练。
- 目的：验证模型具备拟合数据的能力；
- 判断标准：若无法在小样本上做到100%训练精度，代表网络结构、学习率、优化器存在bug，无法正常学习。

### Step 3: Find LR that makes loss go down 找到能让损失下降的学习率
遍历多组学习率，快速迭代1~5轮，筛选出损失稳定下降的学习率区间；
学习率过大：损失震荡、不下降；学习率过小：下降极慢，收敛耗时过长。

### Step 4: Coarse grid of hyperparams, train for ~1-5 epochs 
搭建大范围超参网格（学习率、权重衰减、dropout概率、batch size等），每组仅训练1~5个epoch快速缩小候选区间。

### Step 5: Refine grid, train longer 细粒度网格搜索
基于粗筛结果，缩小超参取值范围，使用更密集的参数间隔，完整训练多轮，选出最优超参组合。

### Step 6: Look at loss and accuracy curves 分析训练/验证曲线
训练完成后通过精度曲线判断模型状态（欠拟合/正常/过拟合），针对性调整模型、正则化、数据量。

<img width="585" height="348" alt="Image" src="https://github.com/user-attachments/assets/67c896db-0b6d-4ed6-83e1-7d2683d05efb" />

训练、验证曲线无明显gap，两者同步缓慢上涨，未出现平台；模型还未收敛，训练轮次不足；增加训练epoch，延长训练时间。

<img width="665" height="352" alt="Image" src="https://github.com/user-attachments/assets/7e469d1c-409f-4616-9dd0-084aee585be3" />

训练与验证精度差距持续拉大，验证集精度后期下滑；模型泛化能力差； 增强正则化或者 扩充训练数据集规模。

<img width="645" height="343" alt="Image" src="https://github.com/user-attachments/assets/079cbee1-458e-4a92-a3e5-5860dc781569" />

训练、验证精度都偏低，两条曲线几乎无间隙；模型容量不足，无法捕捉数据特征；
 延长训练轮次， 增大模型规模（加深网络、增加通道数）或者适当调大学习率，减少正则化强度。