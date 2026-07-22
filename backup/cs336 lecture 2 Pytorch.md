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
