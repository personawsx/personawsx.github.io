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

### （4）Vanilla(-ish) RNN: Concrete Example
#### 任务说明：
Manually creating a recurrent network for detecting repeated 1s
手动构建循环网络，检测序列中连续出现的两个1

### 核心公式（本案例使用ReLU激活）

$$
h_t = \text{ReLU}(W_{hh}h_{t-1} + W_{xh}x_t)
$$

$$
y_t = \text{ReLU}(W_{hy}h_t)
$$

隐藏向量结构：

$$
h_t = \begin{pmatrix}
\text{Current} \\
\text{Previous} \\
1
\end{pmatrix}
$$

含义：第1维存当前输入、第2维存上一步输入、第3维为常数偏置1

权重矩阵定义代码
```python
# 输入x -> 隐藏层权重
w_xh = np.array([[1], [0], [0]])

# 隐藏层自循环权重（传递历史状态）
w_hh = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# 隐藏层 -> 输出层权重（红框标注）
w_yh = np.array([1, 1, -1])
# 内部计算等价：Max(Current + Previous - 1, 0)

# 待处理0/1输入序列
x_seq = [0, 1, 0, 1, 1, 1, 0, 1, 1]

# 初始隐藏状态 h_{t-1}
h_t_prev = np.array([[0], [0], [1]])
```
# 2、怎么计算梯度？
a：多对多

<img width="1002" height="521" alt="Image" src="https://github.com/user-attachments/assets/374778d0-c516-46a4-9e6c-517a745f4fc9" />

b：多对一（有时使用最终隐藏状态，有时使用每个时间步的隐藏状态）

<img width="990" height="500" alt="Image" src="https://github.com/user-attachments/assets/bcc992e1-61bb-4f14-87a5-ea765308a9a7" />

c：一对多 （后续每个时间步的输入 = 上一步输出 $$y_{t-1}$$，有时可以设为定值）

<img width="910" height="498" alt="Image" src="https://github.com/user-attachments/assets/45d68ea0-1cee-4e19-aab1-d1790f23519c" />

### Problem：
当序列过长时，会导致GPU内存不足。

### Solution：Truncated Backpropagation through time（截断时序反向传播 Truncated BPTT）
前向：整条序列从头到尾全部计算，前面时序的隐藏状态持续向后传递，不丢弃上下文；
反向：梯度仅在当前窗口内向左传播，窗口左侧更早时间步不参与本轮权重更新。
#### （1）多对一（单输出，仅全局末尾 1 个 Loss）
Loss 仅存在整条序列最后一步；
梯度从末尾唯一 Loss 向左回传，最多传窗口长度步数；
只有最后一段窗口提供梯度更新权重，前面时序无梯度贡献。
#### （2）多对多（逐时刻输出，每步独立 Loss）
窗口内每个时间步都有独立 Loss，最终多个叠加；
梯度仅在当前窗口内部反向计算，窗口左侧时序梯度截断丢弃。

# 3、RNN tradeoffs
## RNN Advantages:
- Can process any length of the input (no context length)
- Computation for step $t$ can (in theory) use information from many steps back
- Model size does not increase for longer input
- The same weights are applied on every timestep, so there is symmetry in how inputs are processed.

## RNN Disadvantages:
- Recurrent computation is slow
- In practice, difficult to access information from many steps back

# 4、RNN Variants: Long Short Term Memory (LSTM)
## 1. Vanilla RNN

$$
h_t = \tanh\left( W \begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix} \right)
$$

## 2. LSTM 
### 四元门控向量计算

$$
\begin{pmatrix}
i \\ f \\ o \\ g
\end{pmatrix}
=
\begin{pmatrix}
\sigma \\
\sigma \\
\sigma \\
\tanh
\end{pmatrix}
W
\begin{pmatrix}
h_{t-1} \\ x_t
\end{pmatrix}
$$

### 细胞状态更新（记忆传送带）
$$
c_t = f \odot c_{t-1} + i \odot g
$$

### 当前时刻隐藏状态输出
$$
h_t = o \odot \tanh(c_t)
$$

### 符号说明
- $i$：输入门（Input gate）
- $f$：遗忘门（Forget gate）
- $o$：输出门（Output gate）
- $g$：候选细胞状态（候选记忆）
- $\sigma$：Sigmoid激活函数，输出0~1权重控制信息流通
- $\odot$：逐元素哈达玛积（向量对应位置相乘）
- $c_t$：LSTM细胞状态（长期记忆通道）
- $h_t$：当前时刻隐藏输出（短期记忆）
- $h_{t-1}$：上一时刻隐藏状态
- $x_t$：当前时刻输入