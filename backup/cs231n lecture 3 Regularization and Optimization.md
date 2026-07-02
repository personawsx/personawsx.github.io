# 1、Regularization 正则化（Prevent the model from doing too well on training data）
### （1）损失函数公式
$$
L(W) = \frac{1}{N}\sum_{i=1}^N L_i \big( f(x_i, W), y_i \big) + \lambda R(W)
$$

### 公式分项说明
### Data loss: Model predictions should match training data
### 数据损失：约束模型预测结果贴合训练样本真实标签
$\frac{1}{N}\sum_{i=1}^N L_i\big(f(x_i, W), y_i\big)$

### Regularization: Prevent the model from doing too well on training data
### 正则项：避免模型在训练集上过拟合
$\lambda R(W)$

正则化的要点：不要过度拟合数据，有时简单的模型反而效果更好

### （2）Simple examples

### L1 regularization：
$$
R(W) = \sum_k \sum_l | W_{k,l} |
$$

其中  $W_{k,l}$是在对输入的值做运算时用到的矩阵 $W_{k,l}$。
### L2 regularization：
$$
R(W) = \sum_k \sum_l W_{k,l}^2
$$

L1偏好稀疏向量，L2偏好稠密、数值较小的向量。

### （3）结合Softmax
### Softmax 单样本交叉熵损失

$$
L_i = -\log\left( \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right) \quad \text{Softmax}
$$
### 完整总损失 Full loss
$$
L = \frac{1}{N}\sum_{i=1}^N L_i + \lambda R(W) \quad \text{Full loss}
$$

# 2、Optimization （How to find the best W）
## 损失曲面 / 损失景观（Loss Landscape）直观解释
## （1） 坐标轴对应关系
**X 轴、Y 轴**：模型的权重参数 $W$
  如果网络有成千上万个权重，现实是超高维空间，图里只用 2 根轴简化代表任意两个参数；
  你每一步调整权重 $W_{k,l}$，就相当于在 xy 平面里移动坐标点。

**Z 轴（高度）**：总损失 $L(W)$ 的数值；
  高度越高 = 损失越大，模型效果越差；高度越低 = 损失越小，模型拟合效果越好。