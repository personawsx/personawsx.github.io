# 1、Linear attention
## 1. Q/K/V 定义
- $Q\in\mathbb{R}^{n\times d_k}$：Query，查询矩阵，每行代表单个token的查询向量
- $K\in\mathbb{R}^{n\times d_k}$：Key，键矩阵，每行代表单个token的检索特征
- $V\in\mathbb{R}^{n\times d_v}$：Value，值矩阵，每行代表单个token携带的内容信息
- $n$：序列token长度； $d_k$：Q/K维度； $d_v$：V维度 
这里的QKV是利用输入向量乘以不同的矩阵得到的。

## 2. 标准注意力基础形式
$$Attn(Q,K,V) = \rho(QK^\top)V$$
$\rho$ 一般代表 Softmax。
原始计算顺序 $(QK^\top)V$：
1. 先计算 $QK^\top$，生成 $n\times n$ 相似度矩阵；
2. 算力复杂度 $\boldsymbol{O(n^2 d_k + n^2 d_v)}$，**与序列长度n呈平方关系**，长文本开销极高。

## 3. 核心数学思想：矩阵乘法结合律
在 **暂时移除Softmax（ $\rho$为恒等映射）** 前提下：
$$(QK^\top)V = Q(K^\top V)$$
- 原版顺序：先两两匹配所有Token，产生平方复杂度
- 调换运算顺序：优先计算 $K^\top V$
新复杂度 $\boldsymbol{O(2nd_k d_v)}$，**和序列长度n呈线性关系**，降低了复杂度。

# 2、Recurrent form（递归形式） of linear attention
### 2.1 并行矩阵形式 $(QK^\top)V$
训练时拥有整条完整输入序列，全部token同时可用
充分利用显卡并行算力，训练速度快，但需要加载全部序列，不适合逐token串行生成任务

### 2.2 串行递归（RNN式）形式
$$S_t = S_{t-1}+k_tv_t^\top,\quad y_t=q_t^\top S_t$$
维护固定大小的状态矩阵 $S_t$，不断累积历史token的 $k_tv_t^\top$
无需缓存全部历史K/V，内存不随序列长度持续膨胀；线性复杂度低，长文本生成更高效
这里的t对应的是第t个token。

# 3、From linear attention to Mamba-2

<img width="824" height="485" alt="Image" src="https://github.com/user-attachments/assets/de5aaf33-6ea9-4ba8-82cf-5b0ddf6bb5af" />

线性注意力太简单——作小调整

# 4、Gated delta net (and friends)

<img width="850" height="327" alt="Image" src="https://github.com/user-attachments/assets/038e83d8-6792-43fe-bb1f-2232bdcd6f73" />

# 5、Alternative to hybrids: sparse adaptation

<img width="1047" height="632" alt="Image" src="https://github.com/user-attachments/assets/bbeb656d-65ed-4a5c-b0a8-9741b0f17d78" />

先用一个超轻快的小网络快速粗筛出最相关的token，只对筛选后的少量token执行正式注意力计算，抛弃无关token，以此来降低复杂度。

# 6、MoE

<img width="1306" height="586" alt="Image" src="https://github.com/user-attachments/assets/c5b7d2fa-ebef-4fe0-8586-c227bc894287" />

有多个不同的FNN来处理不同的语义
为什么说MoE是稀疏的？
答：稀疏 = 执行一次前向传播时，只用到模型全部参数里的一小部分，剩下大部分参数不参与计算。MoE有多个FFN，处理某一token时只会调用其中的一个，所以说MoE是稀疏的。

<img width="932" height="732" alt="Image" src="https://github.com/user-attachments/assets/d9f909eb-bf30-4c91-aba2-dfeb8f4a334b" />

# 7、MoE-What Varies？
## 7.1 Routing function 

<img width="1065" height="627" alt="Image" src="https://github.com/user-attachments/assets/ffdd52f1-f737-4e6a-a982-869e3e76ce01" />

行：token
列：expert
格子：分数

Token chooses expert：可能有多个expert被复用，会导致一些expert闲置
Expert chooses token：一个token可能被多个expert选中

Almost all the MoEs do a standard ‘token choice topk’ routing.

<img width="1407" height="766" alt="Image" src="https://github.com/user-attachments/assets/650dab86-3596-4e32-abb2-d0477664c9a6" />

<img width="1485" height="754" alt="Image" src="https://github.com/user-attachments/assets/bb3a84a5-05fd-4a21-be0c-42715c7a7fc7" />

# Top‑K routing in detail

### 1）输出计算公式

$$
\mathbf{h}_t^l = \sum_{i=1}^N \Big(g_{i,t}\, \text{FFN}_i\big(\mathbf{u}_t^l\big)\Big) + \mathbf{u}_t^l
$$

- $t$：代表第 $t$ 个token；$l$：第 $l$ 层MoE层
- $\mathbf u_t^l$：该层输入token特征向量
- $N$：专家总数量；$\text{FFN}_i$：第 $i$ 个专家（前馈网络）
- $g_{i,t}$：**gate门控权重**，0代表不选择该专家；非0代表选中，权重参与加权求和
- 最后 $+\mathbf u_t^l$ ：残差连接，和Transformer标准残差完全一致。

> 含义：把token送给选中的专家FFN计算，结果用门控权重加权求和，再加上原始输入残差，得到层输出。

### 2）门控权重 $g_{i,t}$ 定义

$$
g_{i,t}=
\begin{cases}
s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t}|1\le j \le N\},K) \\
0, & \text{otherwise}
\end{cases}
$$

- 对token $t$，取出所有专家分数 $s_{*,t}$ ，选出分数最高的 $K$ 个专家。
- 如果专家 $i$ 属于top‑k集合， $g_{i,t}=s_{i,t}$ ；其余专家直接置0，不参与计算。

### 3）专家分数 $s_{i,t}$（DeepSeek v1‑2 / Qwen / Grok 版本）

$$
s_{i,t}= \text{Softmax}_i\Big({\mathbf u_t^l}^\mathrm{T}\mathbf e_i^l\Big)
$$

- $\mathbf e_i^l$：门控网络的可学习参数向量（每个专家对应一个向量）。
- $\mathbf u_t^l{}^\mathrm{T}\mathbf e_i^l$：输入token向量和专家向量做点积，得到原始打分。
- $\boldsymbol{\text{Softmax}_i}$： **先对全部N个专家做Softmax归一化，再取Top‑K** 。

<!-- Failed to upload "image.png" -->

Deepseek的创新：引入了共享专家，使token不需要经过路由也可输入，同时将输入路由的token分配到不同的专家。

How do we train MoEs?
Major challenge: we need sparsity for training‑time efficiency…
But sparse gating decisions are not differentiable!
Solutions?
Reinforcement learning to optimize gating policies（强化学习）
Stochastic perturbations（随机扰动）
Heuristic ‘balancing’ losses.（负载均衡，常用⭐）
Guess which one people use in practice?

## 负载均衡

<img width="1276" height="697" alt="Image" src="https://github.com/user-attachments/assets/4401ddfe-7b2f-49d3-aa9a-fc192799c92d" />

<img width="999" height="713" alt="Image" src="https://github.com/user-attachments/assets/ee668055-470c-4344-8605-c03bbc9298d9" />

<img width="1211" height="715" alt="Image" src="https://github.com/user-attachments/assets/5dbc0ac6-d92a-4075-b0c0-a6b5f8ec3d68" />

## 7.2 Experrt sizes


## 7.3Training objectives
