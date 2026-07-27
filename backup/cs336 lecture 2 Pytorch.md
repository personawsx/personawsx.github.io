Recall: what's the best model one can train given fixed resources (compute, memory)?
In other words: maximize (computational) efficiency.
Prerequisite: understand the resources (compute, memory) for a given computation.

motivating_questions()
What knowledge to take away from this lecture:    
Mechanics: straightforward (PyTorch semantics) 
Mindset: resource accounting (remember to do it)   
Intuitions: get a sense of how resources are spent, no ML magic today
 
### 1、def tensors_basics():（张量的定义）
    Tensors are the basic building block for storing everything:
data    
parameters 
gradients  
optimizer state   
activations
张量包括向量、矩阵甚至任意阶的数组

问：张量需要多少存储空间？答：取决于张量的类型，一般张量里存储的是浮点数（大多指float32），也可以为整数或者其他类型。

### float32

<img width="2064" height="262" alt="Image" src="https://github.com/user-attachments/assets/2fba7af3-0616-49cc-a6f9-efe0826012a1" />

单个float32的元素占4字节（8位一字节）

### float16

<img width="1081" height="281" alt="Image" src="https://github.com/user-attachments/assets/0fb8484d-256a-457b-a0b5-65ea5a8f2f28" />

The fp16 data type (also known as float16 or half precision) cuts down the memory.
However, the dynamic range (especially for small numbers) isn't great.

### bf16

<img width="1096" height="277" alt="Image" src="https://github.com/user-attachments/assets/001c767e-5109-45e6-a2de-e9f1e402e86e" />

Google Brain developed brain floating point (bf16) in 2018 to address this issue.
bf16 uses the same memory as fp16 but has the same dynamic range as fp32!
The only catch is that the resolution is worse, but this matters less for deep learning.

### Mixed precision
Implications on training:   
Training with fp32 works, but requires lots of memory.
Training with fp16 and even bf16 is risky, and you can get instability.
   
#### Solution: 
mixed precision training  [[Micikevicius+ 2017]](https://arxiv.org/pdf/1710.03740.pdf)  
Use bf16 for parameters, activations, and gradients  
Use fp32 for optimizer states

Pytorch has an automatic mixed precision (AMP) library.  [[docs]](https://pytorch.org/docs/stable/amp.html)
Tries to cast things into bf16 when safe (matmuls, not exp).

<img width="850" height="273" alt="Image" src="https://github.com/user-attachments/assets/c9a74d2f-5803-4f07-a157-530cab495fdb" />

### einops
### （1）einsum
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  
Dimensions that are not named in the output are summed over.
#### Or can use `...` to represent broadcasting over any number of dimensions
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")  

### （2）reduce
You can reduce a single tensor via some operation (e.g., sum, mean, max, min).
x = torch.ones(2, 3, 4)  # batch seq hidden 
#### Old way
y = x.sum(dim=-1)  
#### New (einops) way
y = reduce(x, "... hidden -> ...", "sum")  

### （3）rearrange
Sometimes, a dimension represents two dimensions and you want to operate on one of them.
下面是一个先拆分再合并的例子：
x = torch.ones(3, 8)  # seq total_hidden 
...where total_hidden is a flattened representation of heads * hidden1
w = torch.ones(4, 4)  # hidden1 hidden2 
### Break up `total_hidden` into two dimensions (`heads` and `hidden1`
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)  
### Perform the transformation by `w`
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")  
### Combine `heads` and `hidden2` back together
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")  

# tensor_operations_flops
A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x y).

Two terribly confusing acronyms (pronounced the same!):  
FLOPs: floating-point operations (measure of computation done) 
FLOP/s: floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware.

# Linear model
B：Number of points
D：Dimension of each point
K：Number of outputs

x = torch.ones(B, D, device=cuda_if_available())
w = torch.randn(D, K, device=cuda_if_available())
y = x @ w

How many FLOPs is this matmul?
We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.
actual_num_flops = 2 * B * D * K  

注：矩阵计算每一次乘法和加法算一次flop，加法原本应该是D-1，但是为了计算，统一记作D，故结果为2D。

We can also time this operation to see how long it takes.
actual_time = benchmark(lambda: x @ w)  

# Model FLOPs utilization (MFU)
表示实际FLOPS与纸面FLOPS之间的差异
Definition: MFU = (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]
mfu = actual_flop_per_sec / promised_flop_per_sec if promised_flop_per_sec else None  
Usually, MFU of ≥ 0.5 is quite good!
But why is MFU not closer to 1?
To answer this question, we need to look more closely at how computations are done on GPUs...

# Summary
Matrix multiplications dominate: (2 m n p) FLOPs
FLOP/s depends on hardware (B200 >> H100) and data type (bfloat16 >> float32)
Model FLOPs utilization (MFU): (actual FLOP/s) / (promised FLOP/s)

# arithmetic_intensity：

<img width="709" height="621" alt="Image" src="https://github.com/user-attachments/assets/5c70b085-dbdf-49d2-b332-e0b3fdd6de3b" />

注：图片中的compute就是指一般意义上的加速器，如GPU等

How to compute a thing:
    
Send inputs from memory to accelerator    
Perform computation   
Send outputs from accelerator to memory
    
How long does this take?    
Depends on two things:   
Accelerator speed (FLOP/s)  
Memory bandwidth (bytes/s)

communication_time = bytes / bytes_per_sec  
computation_time = flops / flop_per_sec

一般假设计算和通信能够同时进行，此时total_time=max(communication_time, computation_time) 

What is the bottleneck?  
Memory-bound: communication time > computation time
Compute-bound: computation time > communication time

Alternative way to see this:

Accelerator intensity: how much work can the accelerator do per byte transferred?（由硬件决定）
example：h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  
    
Arithmetic intensity: how much actual work per byte for this workload?（由算法决定）
arithmetic_intensity = flops / bytes 
     
What is the bottleneck?   
Memory-bound: arithmetic intensity < accelerator intensity    
Compute-bound: arithmetic intensity > accelerator intensity

一般的算法大多是memory-bound，as long as we have large matrices, we're compute-bound (saturating the accelerator).

# 计算量近似：
对于 MLP 以及短上下文 Transformer的训练（前向 + 反向传播），可以近似：
前向传播：2NP FLOPs
反向传播：4NP FLOPs
合计一轮训练：6NP FLOPs
N=样本数量，
P=网络总参数量。
原理：前向只做一次线性计算；反向传播必须同时求解权重梯度和输入梯度两组梯度，开销是前向的 2 倍。

# Summary:
• Everything is operations on tensors (parameters, gradients, activations, optimizer states, data)
• einops: better way to think about tensor operations
• 6 (# data points) (# parameters) FLOPs per training step
• Arithmetic intensity / roofline analysis: compute-bound or memory-bound?
• Matrix multiplications are compute-bound, elementwise operations are memory-bound
• Gradient accumulation, activation checkpointing: reduce memory to use bigger batch sizes