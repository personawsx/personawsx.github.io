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

由于Top-K MoE中的专家负载不均衡（有的专家被多个token选中，有的专家不会被token选中，会造成浪费），因此采用负载均衡。
#### （1）Switch Transformer 原始均衡损失（Fedus 2022，K=1，每个 token 只选 1 个专家）

$$
\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i
$$

$$
f_i = \frac{1}{T}\sum_{x\in \mathcal{B}} \mathbb{1}\{\text{argmax }p(x)=i\}
$$

$$
P_i = \frac{1}{T}\sum_{x\in \mathcal{B}} p_i(x)
$$

变量解析：
1. $N$：专家总数量
2. $\alpha$：超参数，控制这个辅助loss有多强（不能太大，否则会破坏主任务学习）
3. $\mathcal{B}$：一个训练批次 batch； $T$：这个batch里全部token总数
4. $\mathbb{1}\{\dots\}$：**指示函数**，括号内条件成立=1，不成立=0
5. $f_i$：**真实负载（统计值，不可导）**
    遍历batch所有token，统计：最终真正分配给专家 $i$的token占全部token的比例。
6. $p_i(x)$：Gate网络对token $x$，分配给专家$i$的路由概率（Softmax输出，**可导！Gate的参数在这里**）
7. $P_i$：**Gate预测平均概率（可导）**
    整个batch内，Gate分给专家 $i$的路由概率平均值
即：
$f$ 向量 = 每个专家**实际分到多少token**
$P$ 向量 = Gate网络**预测每个专家应该分到多少token概率**
$\sum f_i P_i$ 是两个向量做点积。
优化目标是想最小化这个点积

Switch的局限
只支持 **K=1（每个token仅选1个专家）**，不能直接用于 Mixtral / DeepSeek 这类 K>2 的模型。

#### （2）DeepSeek V1 / V2 改进（支持K>1）

两套独立辅助损失一起加到总loss：
### ① $\mathcal L_{\text{ExpBal}}$ 专家粒度均衡（适配Top-K，K>1，和Switch思想同源）

$$
\mathcal L_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i
$$

$$
f_i = \frac{N'}{K'T}\sum_{t=1}^T \mathbb{1}(\text{Token }t \text{ selects Expert }i)
$$

$$
P_i = \frac{1}{T}\sum_{t=1}^T s_{i,t}
$$

- $N'$：当前层专家总数； $K'$：每个token激活K个专家
- $s_{i,t}$：Gate输出的专家匹配得分（Softmax前/后得分）
- 改动：增加 $\frac{N'}{K'T}$ 归一化系数，适配**一个token同时选K个专家**的场景

### ② $\mathcal L_{\text{DevBal}}$ 设备粒度均衡（DeepSeek新增，均衡不同设备的负载）

$$
\mathcal L_{\text{DevBal}} = \alpha_2 \sum_{i=1}^D f_i' P_i'
$$

现代大模型：专家分散部署在多张GPU上，一张GPU托管多个专家，就算每个专家负载均衡，如果高负载专家全都放在同一张GPU，这张卡就会成为整个训练的速度瓶颈。

- $D$：GPU设备数量
- $\mathcal E_i$：部署在第 $i$ 张卡上的所有专家集合
- $f_i'$：第 $i$ 张GPU上，所有专家的平均真实负载
- $P_i'$：第 $i$ 张GPU上，所有专家Gate路由概率总和

同样做向量点积，最小化loss，约束每张GPU整体负载均衡。

#### （3）DeepSeek V3：Per-expert bias
不再依赖辅助loss去惩罚不均衡，直接修改Gate路由打分，在线动态调节每个专家被选中的概率

#### 路由公式改动

$$
g_{i,t}'=
\begin{cases}
s_{i,t}, & s_{i,t}+b_i \in \text{Topk}(\{s_{j,t}+b_j|1 \le j \le N_r\},K_r),\\
0, & \text{otherwise}.
\end{cases}
$$

$b_i$ = **每个专家独立的可在线更新偏置 bias**
路由流程变成：
1. Gate 算出基础得分 $s_{i,t}$
2. 每个专家加上自己专属偏置： $s_{i,t}+b_i$
3. 基于加偏置后的分数，再执行 Top-K 选择

#### $b_i$ 怎么动态调整？
在线学习规则（逻辑）：
- 如果专家 $i$ 最近被大量Token选中（过载）：降低 $b_i$
  → $s_{i,t}+b_i$ 整体变小，后续更难被选入Top-K
- 如果专家 $i$ 长期闲置：提高 $b_i$
  → $s_{i,t}+b_i$ 整体变大，更容易被Token选中

<img width="1137" height="588" alt="Image" src="https://github.com/user-attachments/assets/a4bea683-29d9-4f25-8cab-f905c3530759" />

相较于普通的transformer模型，MoE多了专家并行。

<img width="1117" height="539" alt="Image" src="https://github.com/user-attachments/assets/c7839c1f-e367-40ec-9516-27007a20af7a" />

专家计算在底层矩阵运算上的三种实现方式：
(A) Batched Matrix Multiplication 批量矩阵乘
每个专家独立做矩阵乘法，输入、专家权重尺寸完全一致，并行计算。
缺点：必须提前固定每个专家最多能处理多少 token，不支持负载不均衡。
(B) Block Diagonal Matrix Multiplication 分块对角矩阵乘
把所有专家权重纵向拼接成一个大矩阵，等价为分块对角矩阵乘法，对角块对应各个专家，其余位置为 0。
优点：可以用标准 GEMM；缺点：仍然强制每个专家输入尺寸相同，不支持 token 数量不一样。
(C) Block Sparse Matrix Multiplication 分块稀疏矩阵乘
支持负载不均衡路由、每个专家处理 token 数量不一样、专家尺寸可变。矩阵大量区域是空白 0，只计算有效非零块。

<img width="1089" height="568" alt="Image" src="https://github.com/user-attachments/assets/8fb2be1d-bcc0-4701-b855-a80ae4944750" />

