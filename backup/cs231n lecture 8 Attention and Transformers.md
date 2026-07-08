# 1、Sequence to sequence  with RNNs：
### 编码器 - 解码器（Encoder-Decoder）模型

<img width="937" height="390" alt="Image" src="https://github.com/user-attachments/assets/2298c0d9-c346-4a8f-a932-95c0cbc58d5a" />

## Encoder 编码器
### 1. 公式
$$h_t = f_W(x_t, h_{t-1})$$
- $f_W$：带权重 $W$ 的 RNN 单元（基础 RNN/LSTM/GRU）
- $x_t$：第 $t$ 个输入词的词嵌入向量，图中输入 $x_1=\text{we},x_2=\text{see},x_3=\text{the},x_4=\text{sky}$
- $h_{t-1}$：上一时刻隐藏状态， $h_t$ 当前时刻隐藏状态
- 时序流向： $x_1 \rightarrow h_1 \rightarrow h_2 \rightarrow h_3 \rightarrow h_4$，信息单向向后传递

### 2. 上下文向量 Context vector $c$
- $T$ 是输入序列总长度（本例 $T=4$）
- 一般直接取编码器最后一步隐藏状态 $h_4$ 作为全局上下文向量 $c$
- 作用：把整句英文的全部语义压缩进单一固定维度向量，传给解码器

## Decoder 解码器
### 1. 公式
$$s_t = g_U(y_{t-1}, s_{t-1}, c)$$
- $g_U$：解码器 RNN 单元（权重 $U$，和编码器权重独立）
输入三部分：
1. $y_{t-1}$：上一步生成的单词（词嵌入）；第一步输入特殊起始符 $y_0=[\text{START}]$
2. $s_{t-1}$：解码器上一时刻隐藏状态
3. $c$：全局上下文向量（每一步解码都会复用同一个 $c$，这是无注意力机制的短板）
- $s_t$：解码器当前隐藏状态，用来预测当前输出单词 $y_t$

### 2. 解码器初始状态 $s_0$
Initial decoder state $s_0$
- 解码器第一个隐藏状态 $s_0$ 直接由上下文向量 $c$ 初始化，让解码器一开始就拥有整句输入的全局语义。

Problem：上下文向量c的规格是固定的，对于较短的输入序列效果可能好，但当输入序列较长时，可能会影响解码效果。
Solution: Look back at the whole input sequence on each step of the output(在图中指每一个意大利单词生成的时刻)

# 2、Sequence to sequence  with RNNs and Attention

<img width="982" height="445" alt="Image" src="https://github.com/user-attachments/assets/f05489a2-d287-4fd5-a620-f7827ac7b1bc" />

<img width="977" height="510" alt="Image" src="https://github.com/user-attachments/assets/7eb66fd0-2034-4373-bfe0-ad8763db5592" />
