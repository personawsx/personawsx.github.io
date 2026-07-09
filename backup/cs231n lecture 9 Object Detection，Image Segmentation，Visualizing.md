# 1 ViT（Vision Transformer）
## 1.1 使用cls token
<img width="1003" height="505" alt="Image" src="https://github.com/user-attachments/assets/28bbf6fb-5aee-4655-b940-c3a7396426de" />

## （1）基础概念回顾
1. **Patch**：单张图像分割出的像素小方块；
2. **N**：单张图片分割得到的Patch总数量，图片所示维9；
3. **Token**：经过线性投影后的D维特征向量，是Transformer处理的最小序列单元；
4. **D**：单个Token的特征向量维度；
5. **L**：单张图完整序列长度， $L=N+1$（N个Patch Token + 1个分类Token）；
6. **Batch**：模型一次并行处理的图片总数；
7. **NLP Transformer**：Transformer最初用于自然语言处理（见lecture 8），ViT直接复用完全相同的Encoder结构。

## （2）ViT 完整分步流程
### 步骤1：图像切分，生成原始Patch
将RGB原图均匀切割为无重叠的固定尺寸像素块，单张图得到 $N$个像素Patch。注意此时仅为像素数组，还不是Token。

### 步骤2：线性投影，将Patch 转换为 Patch Token
对每个16×16×3的像素Patch做展平，通过线性全连接层做矩阵映射，输出固定长度**D维特征向量**。
> 关键：这一步完成「像素Patch → 向量Token」的转换。

### 步骤3：叠加位置编码
将上一步得到的D维Patch向量，与同维度可学习的位置编码向量逐元素相加。解决Transformer无法天然感知图像块空间位置的缺陷。

### 步骤4：添加分类Token
在所有Patch Token序列最前端，拼接1个独立可学习的D维**分类Token（cls token）**。
此时输入张量形状：`[batch, L, D]`
- batch：单次并行图片数量
- L：单张图Token总个数 $L=N+1$
- D：每个Token的特征维度

### 步骤5：送入Transformer Encoder
输入与NLP结构完全一致的Transformer编码器。
Transformer输入输出维度不变，输出张量仍为 `[batch, L, D]`。

### 步骤6：单独提取分类Token作为全局图像特征
Transformer会输出一整串等长Token向量，**仅保留每张图头部的分类Token输出向量**，丢弃所有Patch对应的输出向量。
提取后张量形状简化为：`[batch, D]`

### 步骤7：线性分类头映射，输出分类预测分数
将D维分类Token向量送入单层线性层，线性层本质为「矩阵乘法+可学习偏置」：
把D维特征向量映射为C维向量（C=分类总类别数），得到每一类的预测得分；
搭配Softmax即可输出各类别概率，完成图像分类任务。

## 1.2 用Pooling Layer（池化层）代替cls token
<img width="985" height="513" alt="Image" src="https://github.com/user-attachments/assets/62610efa-bba9-481a-9f71-056c0f19e96c" />

# 2、Tweaking Transformers
### （1）：
<img width="451" height="502" alt="Image" src="https://github.com/user-attachments/assets/8fd5d5d2-3065-40c0-9375-d880ed6add61" />

<img width="442" height="492" alt="Image" src="https://github.com/user-attachments/assets/0ac25fd5-afe8-4dac-94c5-d156919e63d4" />
Problem：Layer normalization is outside the residual connections.Kind of weird, the model can’t actually learn the identify function.
Solution: Move layer normalization before the Self-Attention and MLP, inside the residual connections. Training is more stable.

### Post-LN（后置 LN）为什么学不会恒等映射？
第一层输出：
$$X_{out} =\text{LN}(X_{in} +\text{SelfAtt}(X_{in}))$$

假设模型想学习恒等映射（注意力模块输出全 0）：
此时 $\text{SelfAtt}(X_{in})=0$，式子变为：
$$X_{out} =\text{LN}(X_{in})$$

LN 会强制把 $X_{in}$ 标准化到均值 0、方差 1，输出永远不等于原始输入 $X_{in}$。


### Pre-LN（前置 LN）怎么解决这个问题？
第一层输出：
$$X_{mid} =X_{in} +\text{SelfAtt}(\text{LN}(X_{in}))$$

当模型想做恒等映射时，只需让 $\text{SelfAtt}(\cdot)=0$，直接得到：
$$X_{mid} =X_{in}$$

LN 只作用在支路，残差直通路完全保留原始输入，模型可以实现恒等映射。

### （2）全连接层优化：
<img width="973" height="500" alt="Image" src="https://github.com/user-attachments/assets/002b8078-ffca-4d99-8798-162ee047c9d2" />

Classic MLP:

$Y = \sigma(XW_1)W_2$

SwiGLU MLP:

$Y = (\sigma(XW_1) \odot XW_2)W_3$

### （3）Mixture of Experts（MoE）

<img width="986" height="502" alt="Image" src="https://github.com/user-attachments/assets/1b618e98-b638-4ae8-bbc5-59f57a38e7a0" />

MoE是对 Transformer 中 MLP 模块的升级，每层搭建 E 套独立的 MLP 权重大幅扩充模型总参数量，再通过路由机制让每个 token 仅分配给远少于专家总数的 A 个专家参与计算，实现模型知识容量大幅提升的同时仅小幅增加实时计算量。

# 3、Computer Vision Tasks
# 3.1 Semantic Segmentation 

<img width="967" height="487" alt="Image" src="https://github.com/user-attachments/assets/8c069b17-8884-4a4b-a45d-80fb625b980d" />
用固定尺寸窗口在整张图逐像素滑动，每截取一块局部 patch，送入 CNN 只预测窗口中心像素的类别，遍历所有像素得到分割图。
Problem：重叠窗口会重复提取大量重复像素特征

<img width="940" height="495" alt="Image" src="https://github.com/user-attachments/assets/8ac582e0-0c53-476c-a119-865ecf129386" />
直接把整张图片送入CNN 提取全局特征，再做像素分割
Problem：语义分割要求输出分割图和原图长宽完全一致

<img width="1001" height="492" alt="Image" src="https://github.com/user-attachments/assets/f7f397fa-9f3a-4796-b99a-080068660952" />
全程只用卷积、不做任何下采样，特征图始终保持原图 H×W 尺寸
Problem：全程高分辨率卷积，特征图尺寸巨大，开销爆炸

<img width="992" height="503" alt="Image" src="https://github.com/user-attachments/assets/9d2e9eb1-04e1-4346-b51e-c76146f5ec01" />
#### 下采样压缩（编码器）
通过池化、步幅卷积逐步缩小特征图，大幅降低计算量
#### 上采样还原（解码器）
对低分辨率高层特征做上采样，把尺寸恢复到输入原图大小
Problem：How to Upsampling？

<img width="955" height="366" alt="Image" src="https://github.com/user-attachments/assets/80d0c0b1-0f06-4c6a-a86e-dd3891604b3d" />

<img width="946" height="485" alt="Image" src="https://github.com/user-attachments/assets/3ebfda2a-5588-4f78-9a7e-127f56cc7d3d" />
如果采用了最大池化，并且保留了位置信息，Max Unpooling可以还原出其所在位置。

## 3.2 Objection Detection
语义分割只按类别区分像素，不会区分个体
<img width="961" height="448" alt="Image" src="https://github.com/user-attachments/assets/bfe8b9a5-9837-4c37-b239-57eb8c429bc4" />
