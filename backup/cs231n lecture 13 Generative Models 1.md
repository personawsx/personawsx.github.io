# 1、Supervised vs Unsupervised Learning 监督学习 vs 无监督学习
### Supervised Learning 监督学习
1. Data: $(x, y)$
   $x$ is data, $y$ is label
2. Goal: Learn a function to map $x \rightarrow y$
3. Examples: Classification, regression, object detection, semantic segmentation, image captioning, etc.

### Unsupervised Learning 无监督学习
1. Data: $x$
   Just data, no labels!
2. Goal: Learn hidden structure in data
3. Examples: Clustering（聚类）, dimensionality reduction（降维）, density estimation（密度估计）, etc.

# 2、Generative vs Discriminative Models

## （1）Discriminative Model（判别模型）:
Learn a probability distribution $p(y|x)$

<img width="607" height="417" alt="Image" src="https://github.com/user-attachments/assets/a17f03a6-239c-4a4f-8c04-49bd6b8e1f2d" />

Problem：无法处理有问题的输入，依旧会给出所有可能的概率分布（即使输入不属于所有标签中的任意一个）

## （2）Generative Model:
Learn a probability distribution $p(x)$

<img width="602" height="452" alt="Image" src="https://github.com/user-attachments/assets/c1927147-7230-44d6-989c-d6b717b96615" />

目标：搞清楚 “什么样的图片是现实中合理、常见的”

## （3）Conditional Generative Model: Learn $p(x|y)$

<img width="623" height="428" alt="Image" src="https://github.com/user-attachments/assets/1485fc8a-e488-41dd-b0ec-82e27266ec4f" />

目标：学习每一类标签专属的图像分布

<img width="931" height="437" alt="Image" src="https://github.com/user-attachments/assets/c932f7df-943b-4645-8a83-d79545968c4f" />

一般来说，Generative model在实际中应用较少，Conditional generative models应用更加广泛

# 3、Why Generative Models?

Modeling ambiguity: If there are many possible outputs x for an input y, we want to model $P(x | y)$

Language Modeling: Produce output text x from input text y

# 4、Taxonomy of Generative Models

<img width="997" height="455" alt="Image" src="https://github.com/user-attachments/assets/e401078d-d694-44ef-b3c3-6709a87259b9" />

左边分支：可以写出概率分布；右边分支：写不出概率分布
左左：精确可计算密度；左右：近似密度
右左：可以一步直接采样；右右：间接迭代采样，需多轮

# 5、Autoregressive Models

<img width="943" height="492" alt="Image" src="https://github.com/user-attachments/assets/13e96a15-fc22-4109-8da4-c7da5a3ac819" />

## 各个参数含义
### 1. $x$：单个数据样本
 $x$ 代表**一条完整训练数据**，根据模型场景含义不同：
大语言模型（GPT自回归）： $x$ = 一整段文本序列 $x=(x_1,x_2,...,x_T)$，  $x_1,x_2$ 是句子里每个单词/Token；
图像模型（PixelCNN自回归）： $x$ = 一张图片，展平成像素序列 $x=(x_1,x_2,...,x_T)$， $x_t$ 是第t个像素；

 $x^{(i)}$ 加了上标 $(i)$：数据集里第 $i$ 个样本，一共有 $N$ 个样本， $i=1,2,...,N$。

### 2. $p(x)$：数据 $x$ 在模型分布下的概率
 $p(x)$ 的完整写法是 $p(x;W)$，代表：**在当前模型参数 $W$ 下，随机生成出样本 $x$ 的概率**。
真实世界存在一个未知的真实数据分布 $p_{\text{data}}(x)$；
我们的神经网络用参数 $W$ 构造一个模型分布 $p(x)=f(x,W)$，去逼近真实分布；
 $p(x)$ 越大 = 模型认为这条样本 $x$ 越“合理、越像真实数据”。

### 3. $W$：模型全部可训练参数

### 4. $f(x,W)$：模型显式概率函数
 $f$ 是神经网络本身，输入样本 $x$、参数 $W$，输出概率值 $p(x)$；
只有**显式密度模型（自回归、VAE）** 能写出这个完整表达式；GAN/扩散做不到，因为它们无法直接输出 $p(x)$。


## MLE核心公式：
$W^{*}=\arg\max_{W}\prod_{i}p(x^{(i)})$

### 1. 前提：样本独立同分布 i.i.d.
 $N$ 条训练样本 $x^{(1)},x^{(2)}...x^{(N)}$ 互不干扰、独立采集。
独立事件联合概率 = 各自概率相乘：

$$
P(\text{同时观测到全部训练样本}) = p(x^{(1)}) \cdot p(x^{(2)}) \cdot ... \cdot p(x^{(N)}) = \prod_{i=1}^N p(x^{(i)})
$$

$\prod_{i}p(x^{(i)})$ 的物理意义：**当前这套参数 $W$，同时生成出所有训练数据的总可能性**。

example：数据集只有2张图片 $x^{(1)},x^{(2)}$
若模型A： $p(x^{(1)})=0.8,\ p(x^{(2)})=0.7$，总可能性 $0.8\times0.7=0.56$
若模型B： $p(x^{(1)})=0.2,\ p(x^{(2)})=0.3$，总可能性 $0.2\times0.3=0.06$
模型A的总概率更大，说明参数A更贴合数据集，MLE就会选A对应的参数。

### 2. $\arg\max_{W}$：找最优参数
$\arg\max_W$ 含义：遍历所有可能的参数 $W$，找到能让后面乘积值最大的那一组参数，记作最优参数 $W^*$，即让全部训练样本同时出现的概率尽可能大**。

### 3. Log Trick：乘积转求和 $\prod p(x^{(i)}) \Rightarrow \sum \log p(x^{(i)})$

### 数学原因1：对数不改变极值位置
$y=\log(z)$ 是严格单调递增函数：
如果 $a>b$，一定有 $\log(a)>\log(b)$。
所以最大化 $\prod p(x^{(i)})$ 和最大化 $\sum \log p(x^{(i)})$，找到的最优 $W$ 完全一样，不会改变训练结果。

### 数学原因2：数值稳定性+梯度计算更简单
1. 数值稳定：大量小数相乘会无限趋近于0，计算机浮点精度直接下溢失效；取log后变成负数相加，不会溢出；
2. 求导简化：乘积求导要用复杂链式法则，加法求导直接分项求和，梯度下降计算量大幅降低。

转换后公式：

$$
W^{*}=\arg\max_{W}\sum_{i=1}^N \log p(x^{(i)})
$$

## 4. 代入模型函数 $p(x)=f(x,W)$，得到最终训练式

$$
W^{*}=\arg\max_{W}\sum_{i=1}^N \log f(x^{(i)},W)
$$

### 损失函数转换
梯度下降算法默认是**最小化损失**，但我们现在目标是最大化对数似然，所以统一取负号：

$$
\text{NLL 负对数似然损失} = -\sum_{i=1}^N \log f(x^{(i)},W)
$$

训练等价于：
$\displaystyle W^{*}=\arg\min_{W}\left[ -\sum_{i=1}^N \log f(x^{(i)},W) \right]$
$\log f(x^{(i)},W)$ 越大，负对数似然损失越小；
模型预测样本概率越高，损失越低，符合训练逻辑。

<img width="957" height="488" alt="Image" src="https://github.com/user-attachments/assets/6e2b4a77-44bf-434b-88e9-86d309323dcc" />

用链式法则写出 $x$的表达式，其与RNN，transformer有些类似

<img width="978" height="497" alt="Image" src="https://github.com/user-attachments/assets/32136bb7-2cc4-4667-bf23-043855570767" />

# 6、Variational Autoencoders（VAEs）

<img width="887" height="465" alt="Image" src="https://github.com/user-attachments/assets/a3e15d0c-e74c-4124-a376-3d371aee725a" />