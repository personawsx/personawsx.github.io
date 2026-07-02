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

# 2、Optimization （How to find the best W？）
## 损失曲面 / 损失景观（Loss Landscape）直观解释
## （1） 坐标轴对应关系
**X 轴、Y 轴**：模型的权重参数 $W$
  如果网络有成千上万个权重，现实是超高维空间，图里只用 2 根轴简化代表任意两个参数；
  你每一步调整权重 $W_{k,l}$，就相当于在 xy 平面里移动坐标点。

**Z 轴（高度）**：总损失 $L(W)$ 的数值；
  高度越高 = 损失越大，模型效果越差；高度越低 = 损失越小，模型拟合效果越好。

## How to find the best W？
（1）Random research
（2）Follow the slope：
In 1-dimension, the derivative（导数） of a function:

$$
\frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

In multiple dimensions：gradient（梯度）
In summary:
Numerical gradient（h版本）: approximate, slow, easy to write
Analytic gradient（损失函数可导）: exact, fast, error-prone

# 3、Gradient descent
### When to stop？
（1）固定迭代次数；
（2）损失不再明显变化。

### Stochastic Gradient Descent (SGD) 随机梯度下降
将数据分为若干 mini-Batch，遍历所有 Batch 称为一个 Epoch

## Problem：
① 学习率过大会冲出山谷
② 学习率不大时，抖动
③ Saddle points in high dimension（极小值点）/梯度为0 时，不会继续下降
④ 对数据集进行子采样 → 存在噪声

### 解决方法：
## （1） SGD + Momentum（动量）
解决问题：③ ④
可以利用动量冲出鞍部，并能抵消一部分噪声

SGD：
$$w_{t+1} = w_t - \alpha \nabla f(w_t)$$

SGD + Momentum：
$$v_{t+1} = \rho v_t + \nabla f(w_t)$$
$$w_{t+1} = w_t - \alpha v_{t+1}$$

```python
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx + dx
    x -= learning_rate * vx
```

## （2）RMSProp

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```
平坦区域步长更大，陡峭区域步长更短

## （3）Adam (almost)
Momentum + RMSProp
```python
first_moment = 0
second_moment = 0
while True:
    dx = compute_gradient(x)
    first_moment = beta1 * first_moment + (1 - beta1) * dx
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
    x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)
```
一般beta2取接近1的值，上述代码中会导致second_moment趋于0，导致开始的步长x极大
优化版本：
```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations):
    dx = compute_gradient(x)
    first_moment = beta1 * first_moment + (1 - beta1) * dx
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
    first_unbias = first_moment / (1 - beta1 ** t)
    second_unbias = second_moment / (1 - beta2 ** t)
    x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)
```
### （4）AdamW（防止正则项干扰梯度下降）
普通Adam在进行梯度下降时包含了正则项部分，但一般希望动量只依赖损失函数，所以在AdamW中，选择将正则项排除在梯度下降的环节，梯度下降完成后，再加上正则项部分。

# 4、Learning Rate Decay
### （1）学习率衰减调度策略
### Step（分段阶梯衰减）
在固定轮次降低学习率。
例：ResNet 标准配置，在第30、60、90轮后，将学习率乘以0.1。

### Cosine（余弦退火衰减）
$$\alpha_t = \frac{1}{2}\alpha_0 \big(1 + \cos(\frac{t\pi}{T})\big)$$

### Linear（线性衰减）
$$\alpha_t = \alpha_0 \big(1 - \frac{t}{T}\big)$$

### Inverse sqrt（平方根倒数衰减）
$$\alpha_t = \frac{\alpha_0}{\sqrt{t}}$$

### 参数说明
- $\alpha_0$：初始学习率
- $\alpha_t$：第 $t$ 轮的学习率
- $T$：总训练轮数（epochs）

### （2）Linear warmup
High initial learning rates can make loss explode; linearly increasing learning rate from 0 over the first ~5,000 iterations can prevent this.