# 1、Recurrent Neural Networks：Process Sequences

<img width="814" height="251" alt="Image" src="https://github.com/user-attachments/assets/3f26d387-c9ff-495d-b1c0-0143e42ae9fd" />

#### RNN特点：当前时刻的计算依赖于当前时刻和前一时刻的RNN状态。

<img width="2163" height="1167" alt="Image" src="https://github.com/user-attachments/assets/83f25cb0-5e08-4196-9bad-d131afbc8ecd" />

## RNN的隐藏状态和输出：
<img width="195" height="371" alt="Image" src="https://github.com/user-attachments/assets/72370c66-5f14-45a3-bdab-0340eef8652f" />

### （1）RNN 隐藏状态更新公式

$$
h_t = f_W\big(h_{t-1},\ x_t\big)
$$

### 各变量/符号注释
- $h_t$：new state，**当前时刻新隐藏状态**
- $f_W$：some function with parameters $W$，带共享参数 $W$ 的变换函数（RNN单元内部计算逻辑，所有时间步共用同一套参数 $W$）
- $h_{t-1}$：old state，**上一时刻旧隐藏状态**（历史记忆）
- $x_t$：input vector at some time step，**当前时间步的输入向量**

### （2）RNN output generation（RNN输出生成）

$$
y_t = f_{W_{hy}}\big(h_t\big)
$$

### 各符号变量注释
- $y_t$（绿色方框，标注output）：当前时间步 $t$ 的模型输出结果
- $f_{W_{hy}}$（紫色方框）：带有参数 $W_{hy}$ 的变换函数，用于将隐藏状态映射到输出空间
- $W_{hy}$：输出层专属权重参数，所有时间步共享该套参数
- $h_t$（蓝色方框，标注new state）：当前时刻计算完成的新隐藏状态，作为输出函数的输入

### 两个公式的关系说明
1. 先通过隐藏状态更新公式算出 $h_t$；
2. 再将 $h_t$ 送入输出函数 $f_{W_{hy}}$，得到当前时刻预测输出 $y_t$；
3. $W_{hy}$ 是独立于隐藏状态更新参数 $W$ 的另一组权重，专门负责隐藏状态到输出的映射（隐藏状态和输出可以维度不同）。

### （3）example
#### (Vanilla) Recurrent Neural Network 标准基础循环神经网络

#### a：Tanh 激活函数
#### 公式
$$
f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

<img width="416" height="275" alt="Image" src="https://github.com/user-attachments/assets/b432f39e-ecee-4e45-b85b-def822953298" />

#### b：核心计算公式
#### 隐藏状态更新公式：

$$
h_t = \tanh\big(W_{hh}h_{t-1} + W_{xh}x_t\big)
$$

符号释义：
- $h_t$：当前时刻 $t$ 的隐藏状态（时序记忆）
- $h_{t-1}$：上一时刻 $t-1$ 的历史隐藏状态
- $x_t$：当前时刻输入向量
- $W_{xh}$：输入层 → 隐藏层权重矩阵
- $W_{hh}$：隐藏层自循环传递权重矩阵
- $\tanh$：非线性激活函数

#### 输出预测公式：

$$
y_t = W_{hy}h_t
$$

注：Often, we also have an output function $f_y$
实际任务中通常会额外增加输出激活函数（如Softmax）。
符号释义：
- $y_t$：当前时刻模型输出
- $W_{hy}$：隐藏层 → 输出层独立权重矩阵





