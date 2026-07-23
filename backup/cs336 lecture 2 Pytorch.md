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

30