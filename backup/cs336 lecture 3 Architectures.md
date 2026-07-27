# 1 Common architecture variations
## 1.1 Pre-vs-post norm

<img width="1196" height="621" alt="Image" src="https://github.com/user-attachments/assets/21f3c497-2dd7-4c2b-a6bc-cb33da7492a9" />

残差主干道保持原始数据，不受归一化干扰，这一点至关重要
若遇到稳定性问题，可以在模型的各个地方适当加上LayerNorm

## 1.2 LayerNorm vs RMSNorm

Original transformer: LayerNorm – normalizes the mean and variance across
 $d_{model}$

$$
y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} * \gamma + \beta
$$

**Notable models:**
GPT3/2/1, OPT, GPT-J, BLOOM

Many modern LMs: RMSNorm – does not subtract mean or add a bias term

$$
y = \frac{x}{\sqrt{||x||_2^2 + \epsilon}} * \gamma
$$

$$\|x\|_2 = \sqrt{x_1^2 + x_2^2 + ... + x_d^2}$$

**Notable models:**
LLaMA-family, PaLM, Chinchilla, T5

### Why RMSNorm?
Modern explanation – it’s faster (and just as good).
- Fewer operations (no mean calculation)
- Fewer parameters (no bias term to store)

$$
y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} * \gamma + \beta
$$

Does this explanation make sense?

| Operator class | % flop |
| ---- | ---- |
| $\triangle$ Tensor contraction | 99.80 |
| $\square$ Stat. normalization | 0.17 |
| $\bigcirc$ Element-wise | 0.03 |

Matrix multiplies are the vast majority of FLOPs (and memory)

Important lesson: FLOPS are not runtime（真实运行耗时）! 

| Operator class | % flop | % Runtime |
| ---- | ---- | ---- |
| $\triangle$ Tensor contraction | 99.80 | 61.0 |
| $\square$ Stat. normalization | 0.17 | 25.5 |
| $\bigcirc$ Element-wise | 0.03 | 13.5 |

RMSNorm can still matter due to the importance of **data movement**

<img width="436" height="488" alt="Image" src="https://github.com/user-attachments/assets/d2a988bb-06d4-45d1-a406-fb2e26245104" />

MHA 拥有 99% 以上的总 FLOPs，但它计算访存比很高，计算效率高；
Dropout、残差相加、LayerNorm 的 FLOPs 加起来很少，但计算访存比很低，全是访存受限，其大量时间消耗在读写显存上。
最终现象：少量 FLOPs 的访存类算子占用非常可观的真实训练时间。

### More generally: dropping bias terms

Most modern transformers don’t have bias terms.

Original Transformer:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

Most implementations (if they’re not gated):

$$
\text{FFN}(x) = \sigma(xW_1)W_2
$$

**Reasons:** memory (similar to RMSnorm) and optimization stability

### LayerNorm: recap
- Basically everyone does non-residual norm (often prenorm)
  - Intuition – keep the good parts of residual connections
  - Observations – nicer gradient propagation, fewer spike
  - Some people add a second norm outside the residual stream

- Most people do RMSnorm
  - In practice, works as well as LayerNorm
  - But, has fewer parameters to move around, which saves on wallclock time
  - People more generally drop bias terms since the compute/param tradeoffs are not great.

## 1.3 Activations

### A whole zoo of activations ..

ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU
What are these things? What do people use? Does it matter?

### A few of the common activations
**ReLU**

$$FF(x) = \max(0, xW_1)W_2$$

<img width="580" height="443" alt="Image" src="https://github.com/user-attachments/assets/b03c5139-2c69-4ae0-9ce5-f9d8f90a207b" />

*Notable models:*
Original transformer, T5, Gopher, Chinchilla, OPT

**GeLU**

$$
\begin{align*}
FF(x) &= \text{GELU}(xW_1)W_2 \\
\text{GELU}(x) &:= x\Phi(x)
\end{align*}
$$

<img width="576" height="422" alt="Image" src="https://github.com/user-attachments/assets/824d2974-efbc-430a-9fc3-54db71ea962f" />

*Notable models:*
GPT1/2/3, GPTJ, GPT-Neox, BLOOM

**SwiGLU / GeGLU (next slide..)**
*Notable models:*
Llama, PaLM, T5 v1.1, most models post 2023

### Gated activations (*GLU)
GLUs modify the ‘first part’ of a FF layer

$$FF(x) = \max(0, xW_1)W_2$$

Instead of a linear + ReLU, augment the above with an (entrywise) linear term

$$\max(0, xW_1) \rightarrow \max(0, xW_1) \otimes (xV)$$

This gives the gated variant (ReGLU) – note that we have an extra parameter $(V)$

$$\text{FF}_\text{ReGLU}(x) = \big(\max(0, xW_1) \otimes xV\big) W_2$$

### Gated variants of standard FF layers

**GeGLU**

$$\text{FFN}_\text{GEGLU}(x, W, V, W_2) = \big(\text{GELU}(xW) \otimes xV\big)W_2$$

*Notable models:*
T5 v1.1, mT5, LaMDA, Phi3, Gemma 2, Gemma 3, Gemma 4

**SwiGLU (swish is $x * \text{sigmoid}(x)$)**

$$\text{FFN}_\text{SwiGLU}(x, W, V, W_2) = \big(\text{Swish}_1(xW) \otimes xV\big)W_2$$

*Notable models:*
LLaMa 1/2/3, PaLM, Mistral, OlMo, most models post 2023

Note: Gated models use smaller dimensions for the $d_{ff}$ by 2/3

## 1.4 Serial vs Parallel layers
Normal transformer blocks are serial – they compute attention, then the MLP
Could we parallelize the transformer block?

# Summary: architectures
### Pre-vs-post norm:
- Everyone does non-residual norm (except OPT350M), likely with good reason.

### Layer vs RMSnorm:
- RMSnorm has clear compute wins, sometimes even performance

### Gating:
- GLUs are consensus now

### Serial vs parallel layers:
- Most models now use serial layers

## 1.5 Many variations in position embeddings

**Sine embeddings**: add sines and cosines that enable localization

$$
\text{Embed}(x, i) = v_x + PE_{pos}
$$

$$
\begin{cases}
PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}}) \\
PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})
\end{cases}
$$

*Notable models:*Original transformer

**Absolute embeddings**: add a position vector to the embedding

$$
\text{Embed}(x, i) = v_x + u_i
$$

*Notable models:*GPT1/2/3, OPT

**Relative embeddings**: add a vector to the attention computation

$$
e_{ij} = \frac{x_iW^Q(x_jW^K + a_{ij}^K)^T}{\sqrt{d_z}}
$$

*Notable models:*T5, Gopher, Chinchilla

**RoPE embeddings**
*Notable models:*GPTJ, PaLM, LLaMA,Most 2024+ models

### RoPE: rotary position embeddings
High level thought process: a relative position embedding should be some $f(x,i)$ s.t.

$$
\langle f(x,i), f(y,j) \rangle = g(x,y,i-j)
$$

That is, the attention function only gets to depend on the relative position $(i-j)$. How do existing embeddings not fulfill this goal?

- **Sine**: Has various cross-terms that are not relative

$$
\langle \text{Embed}(x,i), \text{Embed}(y,i) \rangle = \langle v_x, v_y \rangle + \langle PE_i, v_y \rangle ...
$$

- **Absolute**: obviously not relative

- **Relative embeddings**:

$$
e_{ij} = \frac{x_iW^Q(x_jW^K + a_{ij}^K)^T}{\sqrt{d_z}}
$$
is not an inner product

### RoPE: rotary position embeddings
How can we solve this problem?
- We want our embeddings to be invariant to absolute position
- We know that inner products are invariant to arbitrary rotation.

<img width="1013" height="371" alt="Image" src="https://github.com/user-attachments/assets/52eda7ed-5e32-4fb3-a59c-fb55f3bd8566" />

There are many rotations, which one do you pick?

<img width="1106" height="432" alt="Image" src="https://github.com/user-attachments/assets/2f5235fa-d418-41b6-9aa7-fb794237c6ab" />

Just pair up the coordinates and rotate them in 2d (motivation: complex numbers)
[Su et al 2021]

### 二维角度（复数视角）
把特征向量的**一对维度** $(x_1,x_2)$ 看作复数 $z = x_1 + i x_2$。
复数乘以 $e^{i m \theta}= \cos m\theta + i\sin m\theta$，等价于**平面旋转角度$m\theta$**（ $m$代表token所在位置）。

复数乘法展开：

$$
\begin{align*}
z' &= z \cdot e^{i m \theta} \\
x_1' + i x_2' &= (x_1+i x_2)(\cos m\theta + i\sin m\theta) \\
x_1' &= x_1 \cos(m\theta) - x_2 \sin(m\theta) \\
x_2' &= x_1 \sin(m\theta) + x_2 \cos(m\theta)
\end{align*}
$$

写成矩阵形式：

$$
\begin{pmatrix}x_1'\\x_2'\end{pmatrix}=\begin{pmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}x_1\\x_2\end{pmatrix}
$$

数学性质：
如果 $z_m$ 在位置 $m$旋转角度 $m\theta$， $z_n$ 在位置 $n$旋转角度 $n\theta$
二者内积等价于：原始向量旋转差值角度 $(m-n)\theta$。
**内积只依赖位置差 $m-n$，符合要求**

### The actual RoPE math
Multiply with sines and cosines

<img width="818" height="311" alt="Image" src="https://github.com/user-attachments/assets/9f29f07f-ae72-49a1-9db0-0979c7259a54" />

Difference with sine embeddings – not additive, no cross terms

### 完整流程
输入特征向量： $\boldsymbol{x}_m$ （第 $m$ 个token原始特征）

#### 步骤1：线性投影，得到原始Q、K、V

$$
\begin{align*}
\boldsymbol{q}_m &= \boldsymbol{x}_m W_Q \\
\boldsymbol{k}_m &= \boldsymbol{x}_m W_K \\
\boldsymbol{v}_m &= \boldsymbol{x}_m W_V
\end{align*}
$$

$\boldsymbol{q}_m$：位置 $m$的Query向量
$\boldsymbol{k}_m$：位置 $m$的Key向量
$\boldsymbol{v}_m$：位置 $m$的Value向量

### 步骤2：执行RoPE旋转变换
1. 对 $\boldsymbol{q}_m$：**向量内部维度两两分组，独立二维旋转**，得到旋转后的 $\tilde{\boldsymbol{q}}_m$
$$\tilde{\boldsymbol{q}}_m = R_{\Theta,m}\cdot \boldsymbol{q}_m$$
2. 对 $\boldsymbol{k}_m$：**向量内部维度两两分组，独立二维旋转**，得到旋转后的 $\tilde{\boldsymbol{k}}_m$
$$\tilde{\boldsymbol{k}}_m = R_{\Theta,m}\cdot \boldsymbol{k}_m$$
3. $\boldsymbol{\boldsymbol{v}_m}$：**不做任何变换，直接保留原值**
$$\tilde{\boldsymbol{v}}_m = \boldsymbol{v}_m$$

## 步骤3：计算自注意力

$$
\text{Attention score}_{i,j} = \text{softmax}\left(\frac{\tilde{\boldsymbol{q}}_i \tilde{\boldsymbol{k}}_j^\top}{\sqrt{d}}\right)
$$

$$
\text{Output}_i = \sum_j \text{score}_{i,j}\cdot \tilde{\boldsymbol{v}}_j
$$

即使用旋转后的 $\tilde{q},\tilde{k}$ 计算相似度，乘上原始未旋转的 $v$ 得到输出。

# Implementation and code for RoPE
```python
# Usual attention stuff
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

# Flash attention requires the input to have the shape
# batch_size x seq_length x head_dim x hidden_dim
# therefore we just need to keep the original shape
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

# Get the RoPE matrix cos/sin
cos, sin = self.rotary_emb(value_states, position_ids)
# Multiply query/key inputs
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# Same stuff as the usual multi-head self attention below