# 1、Neural networks
###  （1）the original linear classifier
Linear score function:
$$f = Wx$$
$$x \in \mathbb{R}^D,\ W \in \mathbb{R}^{C \times D}$$
D为输入数据的维度，C为输出数据的维度
### （2）2 layers

2-layer Neural Network
$$f = W_2 \max(0, W_1 x)$$

$$x \in \mathbb{R}^D,\ W_1 \in \mathbb{R}^{H \times D},\ W_2 \in \mathbb{R}^{C \times H}$$
### 注：
max( )是用来建立 $W_1$和 $W_2$之间的非线性关系，此类网络又叫作全连接网络，
max( )又叫作 activation function（激活函数），activation function通常在中间层进行操作。
常见的激活函数有ReLU及其相关变种：

<img width="792" height="525" alt="Image" src="https://github.com/user-attachments/assets/03a86daa-c0bd-4480-a52f-ff766391e58f" />

### 一个两层神经网络的代码实现：

```python

import numpy as np
from numpy.random import randn

# 超参数定义
# N: 样本数量  D_in: 输入特征维度  H: 隐藏层神经元数量  D_out: 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机生成训练输入x、真实标签y
x, y = randn(N, D_in), randn(N, D_out)
# 初始化两层全连接网络权重 w1(输入→隐藏层)、w2(隐藏层→输出层)
w1, w2 = randn(D_in, H), randn(H, D_out)

# 迭代2000轮训练
for t in range(2000):
    # 前向传播：第一层线性变换 + Sigmoid激活函数（引入非线性）
    # z = x.dot(w1) 线性得分，h = σ(z) = 1/(1+e^-z) Sigmoid激活
    h = 1 / (1 + np.exp(-x.dot(w1)))
    # 第二层线性变换，得到模型预测值
    y_pred = h.dot(w2)
    # 损失函数：均方误差MSE，计算预测值与真实标签的平方误差总和
    loss = np.square(y_pred - y).sum()
    # 打印当前迭代轮数与损失值，观察损失下降趋势
    print(t, loss)
   
    # ========== 反向传播 手动求梯度 ==========
    # MSE损失对y_pred的梯度：dLoss/dy_pred = 2*(y_pred - y)
    grad_y_pred = 2.0 * (y_pred - y)
    # 链式求导：损失对w2的梯度 dLoss/dw2 = h^T · grad_y_pred
    grad_w2 = h.T.dot(grad_y_pred)
    # 链式求导：损失对隐藏层h的梯度 dLoss/dh = grad_y_pred · w2^T
    grad_h = grad_y_pred.dot(w2.T)
    # Sigmoid导数 σ'(z)=σ(z)(1-σ(z)) = h*(1-h)，链式求w1梯度
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    # 梯度下降更新权重，学习率设置为1e-4
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2
```
### 注：
neurons数量=中间隐藏层输出向量的维度。
一般不把网络的规模当作正则化中需调整的超参数。

# Problem: How to compute gradients?

### （1）直接求解：
Nonlinear score function（非线性得分函数）：

$$
s = f(x; W_1, W_2) = W_2 \max(0, W_1 x)
$$

Hinge Loss on predictions（铰链损失）：

$$
L_i = \sum_{j \ne y_i} \max(0, s_j - s_{y_i} + 1)
$$

Regularization（L2正则项）：

$$
R(W) = \sum_{k} W_k^2
$$

Total loss: data loss + regularization（总损失 = 数据损失 + 正则损失）：

$$
L = \frac{1}{N}\sum_{i=1}^N L_i + \lambda R(W_1) + \lambda R(W_2)
$$

If we can compute $\displaystyle \frac{\partial L}{\partial W_1},\ \frac{\partial L}{\partial W_2}$ then we can learn $W_1$ and $W_2$
损失函数较为复杂时，上述方法不太可行。

### （2）Computational graphs+Backpropagation（计算图+反向传播）
when Computational graphs are too complex，solution：Backpropagation
<img width="737" height="707" alt="Image" src="https://github.com/user-attachments/assets/76473a48-cc9a-400d-9ffe-483750d1a106" />
反向传播给予了我们获取上游梯度的能力。
图片中输出端下端的1实际上为 $$\frac{\partial f}{\partial x}$$
### example：

<img width="2180" height="822" alt="Image" src="https://github.com/user-attachments/assets/b19bc0ba-6723-4c27-b18f-84f14305e499" />

<img width="1310" height="487" alt="Image" src="https://github.com/user-attachments/assets/2ce1b29a-fa10-4585-9cf3-310fc056e849" />

<img width="1257" height="505" alt="Image" src="https://github.com/user-attachments/assets/c8729ebd-d77c-4031-8d0f-7e8da608e917" />

<img width="1132" height="411" alt="Image" src="https://github.com/user-attachments/assets/58ef2920-ba35-42ae-b3d2-30b10108a680" />
也可以引入sigmoid函数简化化简：
<img width="1422" height="677" alt="Image" src="https://github.com/user-attachments/assets/2927a5a6-bf13-4a29-ab20-5202b5b877c9" />
### Patterns in gradient flow：

<img width="1867" height="1052" alt="Image" src="https://github.com/user-attachments/assets/516f1a67-9f47-4184-a27a-d98df51f4d13" />

### 计算图四大基础门反向梯度规则

### （1）add gate 加法门（梯度分发器）
- 局部梯度恒为1
- 输出梯度原样复制分发到所有输入
例：输出梯度=2，两个输入梯度都为2

### （2）mul gate 乘法门（交换乘数）
- 对a求导局部梯度=b，对b求导局部梯度=a
- 输入梯度 = 上游梯度 × 另一侧输入值
例：上游梯度5，输入2梯度=5×3=15，输入3梯度=5×2=10

### （3）copy gate 复制门（梯度累加器）
- 单输入多输出
- 所有输出支路传回的梯度全部相加，作为输入梯度
例：两路输出梯度4、2，输入梯度=4+2=6

### （4）max gate 最大值门（梯度路由）
- 仅前向取值更大的输入承接全部上游梯度
- 数值更小的输入梯度直接置0
例：max(4,5)=5，上游梯度9，5对应支路梯度=9，4支路梯度=0

### 矩阵/张量反向传播核心要点：

（1）基础前提
损失 L 始终是标量；
$$\frac{\partial L}{\partial X} \text{ 的形状} = \text{原变量 } X \text{ 的形状}$$。
（2）梯度定义
上游梯度 Upstream gradient：∂L/∂z，输出z传回的梯度；
下游梯度 Downstream gradient：∂L/∂x、∂L/∂y，输入的梯度。
（3）多维链式法则
   $$\frac{\partial L}{\partial x} = \frac{\partial z}{\partial x} \frac{\partial L}{\partial z}$$

依靠雅可比矩阵做矩阵乘法完成梯度传递。
（4）核心思想
矩阵/张量反向传播本质和标量链式求导相同，仅计算换成矩阵运算。

