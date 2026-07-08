# 1、Sequence to sequence   with RNNs：
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

<img width="965" height="420" alt="Image" src="https://github.com/user-attachments/assets/74781555-839f-470e-8a00-6368bcebc88f" />

1. 编码器编码输入序列，完整保留所有时序隐向量 $h_1,h_2,...,h_T$；

2. 解码第 $t$步，计算对齐分数： $e_{t,i}=f_{\text{att}}(s_{t-1},h_i)$，标量数值代表当前解码步与第 $i$个输入词的匹配度；
 
3. Softmax归一化分数得到注意力权重 $a_{t,i}$，所有权重之和为1，代表对各输入位置的关注占比；

4. 加权求和编码器全部隐向量，生成当前步专属上下文： $c_t=\sum_i a_{t,i}h_i$；

5. 将动态 $c_t$送入解码器公式  $s_t=g_U(y_{t-1},s_{t-1},c_t)$，更新隐状态并预测输出词；

6. 使用新的 $s_t$重复整套注意力计算，循环生成序列直至终止符。

Query vectors (decoder RNN states) and data vectors (encoder RNN states) get transformed to output vectors (Context states).
Each query attends to all data vectors and gives one output vector

### 1. Query vectors（查询向量）
解码器RNN上一时刻隐藏状态 $s_{t-1}$

### 2. Data vectors（数据向量）
编码器全部时序隐藏状态 $h_1,h_2,...,h_T$

### 3. Output vectors（输出向量）
动态上下文向量 $c_t$（Context states）

一开始计算查询向量与数据向量之间的相似度，即：
 $e_{t,i}=f_{\text{att}}(s_{t-1},h_i)$
得到相似度后，用softmax压缩获得注意力权重，随后求得输出向量，即：
$c_t=\sum_i a_{t,i}h_i$

# 2、Attention Layer
### （1）
<img width="970" height="445" alt="Image" src="https://github.com/user-attachments/assets/1f9d0e8d-0e16-47df-8037-cf803d4067d3" />

这里的 $f_{\text{att}}$ 为取二者的点积并进行缩放，进行缩放的原因是随着维度的增加，可能会出现较高的值对softmax差生影响，使概率分布集中在某一特定区域， $N_X$ 和 $D_Q$ 分别代表对应的维度。

<img width="972" height="496" alt="Image" src="https://github.com/user-attachments/assets/fb7bedde-4a31-4cd5-9f35-142a09634ff2" />
当相同时处理一组查询向量时，查询向量则变为矩阵形式。

### （2）Cross-Attention Layer
在计算相似度和输出向量时都用到了数据向量，为了将这两种用途区分开，引入键和查询的概念：

<img width="987" height="507" alt="Image" src="https://github.com/user-attachments/assets/47233ddb-4c61-4fa8-837d-aab449fa1307" />

### （3）Self-Attention Layer

<img width="978" height="515" alt="Image" src="https://github.com/user-attachments/assets/42cd2a18-f0af-4fc3-b42a-e62e14ce512d" />

#### a:一般处理两种不同类型的数据时，用Cross-Attention Layer，如图像-文字，不同的语言。
#### b:对于Self-Attention Layer，当输入顺序被打乱时，得到的输出相同，但会按照同样顺序被打乱。
Solution：在输入中增加索引。

<img width="956" height="502" alt="Image" src="https://github.com/user-attachments/assets/133cfa02-5139-4f94-999c-2f34f10c292e" />