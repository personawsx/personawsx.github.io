 
<img width="937" height="487" alt="Image" src="https://github.com/user-attachments/assets/01c7141a-26ef-4488-a8d2-52a9a34f9ea9" />
Pretext Objective：
能自动把视觉上同类的图片区分开、聚集在一起，但不能独立完成分类任务，也不认识任何类别名称

# 1、Self-Supervised pretext tasks
<img width="877" height="390" alt="Image" src="https://github.com/user-attachments/assets/5f5f8146-f110-440b-9a6d-30b5603a688b" />

### （1）主要任务：
<img width="883" height="442" alt="Image" src="https://github.com/user-attachments/assets/279cc097-9aa0-4a9f-8723-6e6fb989aeac" />

### （2）How to evaluate？
<img width="903" height="443" alt="Image" src="https://github.com/user-attachments/assets/1f616730-a7e9-49b6-a292-8cd8ca833fdd" />
其中，一般最关注的是下游任务的表现，即Transfer Learning and Downstream Task Performance

<img width="972" height="472" alt="Image" src="https://github.com/user-attachments/assets/964f1664-2fd7-4a8e-99d2-06aa65368650" />

自监督预训练：卷积层 conv + 全连接 fc，以预测旋转角度90°为训练目标，预训练完成后，会丢弃旋转预测用的全连接头，只保留卷积主干用于下游迁移

# 2、Pretext task from image transformations
### （1）predict rotations

<img width="913" height="411" alt="Image" src="https://github.com/user-attachments/assets/ae39ebae-f89e-46d3-8fd0-a4d0989f605a" />

### （2）predict relative patch locations

<img width="845" height="413" alt="Image" src="https://github.com/user-attachments/assets/a7120e06-3a48-4f83-ba03-6e6463476a72" />

 example：

<img width="972" height="408" alt="Image" src="https://github.com/user-attachments/assets/e423bd67-01bf-4f23-aa69-da5c052a7cf1" />
论文提前预先定义了64 种不同的打乱置换规则（表格里index=64代表第 64 套打乱顺序，如9,4,6,8,3,2,5,1,7就是一种 打乱排列）
每一种打乱排列 = 一个独立类别
总共预设了 64 种合法打乱方案，因此输出分类维度是64

### （3）predict missing pixels（inpainting）修复

<img width="828" height="438" alt="Image" src="https://github.com/user-attachments/assets/b920d8c2-30c3-49c5-8f0e-e9f5b1991916" />


总损失由重建损失与对抗损失两部分相加构成：
$$L(x) = L_{recon}(x) + L_{adv}(x)$$
$L_{recon}$ 代表像素重建损失
$L_{adv}$ 代表GAN对抗生成损失

### a:全部符号定义
1. $x$：原始完整真实图像
2. $M$：二值掩码矩阵（与图像同尺寸，逐像素运算）

$$
M_{i,j}=
\begin{cases}
0 & \text{未被遮挡区域（原图保留区域）} \\
1 & \text{被遮挡缺失区域（网络需要补全的区域）}
\end{cases}
$$

3. $*$：逐元素相乘（Element wise multiplication），不是矩阵乘法
4. $F_\theta$：修复生成网络（编码器+解码器，$\theta$ 为网络参数），输入残缺掩码图，输出补全后的完整图像
5. $D$：GAN判别网络，用于区分真实图像与网络修复生成的图像
6. $\|\cdot\|_2^2$：L2平方范数，等价于像素层面均方误差MSE
7. $\mathbb{E}[\cdot]$：对一个批次的图像做数学期望

### b:重建损失 
$L_{recon}(x) = \big\| M * \big(x - F_\theta\big((1-M)*x\big)\big) \big\|_2^2$

#### 分步运算逻辑
1. $(1-M)*x$
掩码取反后与原图逐像素相乘：
- 未遮挡区域 $M=0 \Rightarrow 1-M=1$，保留原图原始像素；
- 遮挡区域 $M=1 \Rightarrow 1-M=0$，像素置零，得到残缺输入图像。
2. $F_\theta\big((1-M)*x\big)$
将残缺图送入生成网络，输出网络修复完成的图像。
3. $x - F_\theta(\dots)$
完整原图减去网络修复图，得到整张图像的像素误差图。
4. $M * \big(x - F_\theta(\dots)\big)$
用原始掩码 $M$ 乘以误差图：**仅保留被遮挡区域的像素误差，完好区域误差直接置0**。
含义：仅约束网络把缺失待修复区域的像素还原到和原图一致，完好可见区域不参与重建损失计算。
5. $\|\cdot\|_2^2$
对遮挡区域所有像素误差做平方求和，即MSE重建损失。

### c:对抗损失 
$L_{adv} = \max_D \mathbb{E}\big[\log(D(x)) + \log\big(1 - D(F((1-M)*x))\big)\big]$

- 判别器 $D$：优化目标为最大化该式 $\max_D$
- 生成器 $F$：整体总损失最小化，等价于最小化 $L_{adv}$

#### 分项含义
1. $\log(D(x))$：输入真实完整图像 $x$ 到判别器，希望判别器输出趋近1，让判别器学会识别真实图像。
2. $\log\big(1 - D(F((1-M)*x))\big)$：输入网络生成的修复假图到判别器，希望判别器输出趋近0，让判别器分辨修复图是生成的假图。

### d:两种损失的意义：
   - 重建损失：约束待修复区域像素和原图匹配；
   - 对抗损失：约束修复区域视觉质感贴近真实图像；

### Pretext task：image coloring

<img width="927" height="473" alt="Image" src="https://github.com/user-attachments/assets/54565dc8-ff7c-441b-8094-78ad42d3c00f" />

 ### 原图拆分两路输入
完整彩色原图 $X$ 拆分成两个独立输入：
- 上分支输入：仅灰度 $L$ 通道（丢失颜色）
- 下分支输入：仅 $ab$ 色彩通道（丢失明暗轮廓）

### 双分支网络互相预测
- 分支网络 $\mathcal{F}_1$：输入灰度 $L$，预测缺失的 $ab$ 色彩图；
- 分支网络 $\mathcal{F}_2$：输入纯色彩 $ab$，反向预测缺失的灰度 $L$ 轮廓图；

### 融合重建完整图像
把 $\mathcal{F}_1$ 预测的 $ab$、 $\mathcal{F}_2$ 预测的 $L$ 拼接，重建出完整彩色预测图像 $\hat{X}$；
两路同时做重建损失约束，双向迫使网络同时学习**空间结构（亮度）**和**语义色彩先验**。

注：RGB 色彩空间无法分离，Lab色彩空间可以分离
 
### 拓展：

<img width="935" height="478" alt="Image" src="https://github.com/user-attachments/assets/db5a3be5-3c03-44d8-aa9d-7b8c93ff1540" />

### （4）video coloring

<img width="982" height="515" alt="Image" src="https://github.com/user-attachments/assets/85f884f2-d045-4a3e-8fa8-5352ca41bd6b" />

输入全部是灰度视频帧，分为两类：
1. **Reference Frame 参考帧**
    视频中拥有真实色彩基准的关键帧，转为灰度后输入CNN；每个像素自带真实颜色标签 $c_i$。
2. **Target Frame 目标帧**
    需要上色的待处理灰度帧，只有轮廓纹理、无颜色信息，是网络需要预测色彩的对象。

### 处理流程
### 1. CNN 提取像素级特征
参考帧、目标帧共用同一个CNN网络，对图像逐像素提取特征向量：
- 参考帧像素特征： $f_i$
- 目标帧像素特征： $f_j$

### 2. 帧间注意力权重计算 $A_{ij}$

$$
A_{ij} = \frac{\exp(f_i^T f_j)}{\sum_k \exp(f_k^T f_j)}
$$

即标准点积自注意力+Softmax归一化：
1. 计算目标像素 $j$ 与参考帧所有像素 $i$ 的特征相似度；
2. 归一化得到权重 $A_{ij}$，代表两帧像素属于同一物体的匹配概率；
3. 示意图黄色箭头代表高权重匹配：两帧中三角形物体特征相似，注意力权重更高。

### 3. 加权预测目标帧像素颜色 $y_j$

$$
y_j = \sum_i A_{ij} c_i
$$

目标帧每个像素的预测颜色，是参考帧所有真实颜色 $c_i$ 按注意力权重加权求和。

### 4. 训练损失函数

$$
\min_\theta \sum_j \mathcal{L}(y_j,c_j)
$$

- $y_j$：网络预测的目标像素色彩
- $c_j$：目标帧像素真实色彩（监督真值）
$\min_\theta$代表最小化所有可学习参数

### （5）MAE（Masked Autoencoder ，掩码自编码器）

<img width="961" height="462" alt="Image" src="https://github.com/user-attachments/assets/cf086225-96b8-46d5-984a-a7c444651913" />

#### 特点：
图像分块逻辑：和原始ViT一致，把完整图片切分为互不重叠的图像块（patch）。
高比例随机掩码：均匀随机遮盖75%的patch，仅保留25%可见patch送入编码器（迫使网络学习图像全局语义）。

<img width="962" height="503" alt="Image" src="https://github.com/user-attachments/assets/3b71bf1f-64cc-47cf-abdf-bd4aa887020f" />

编码器只处理**未被掩码的25%可见patch**

<img width="942" height="408" alt="Image" src="https://github.com/user-attachments/assets/51665aed-2b16-4eed-b46f-a63d61be8db5" />

解码器输入：编码器输出的可见patch特征和**共享可学习掩码token**填充到之前被遮盖的位置，其中全部token统一添加位置编码

<img width="895" height="293" alt="Image" src="https://github.com/user-attachments/assets/53e03c36-2c0b-4b1e-bde2-844b4f1a51c6" />

Linear Probing 线性探测（冻结预训练主干）
将预训练好的 Encoder 全程冻结，在 Encoder 输出的特征后面，只加一层最简单的线性层（全连接层）。训练的时候，只更新这最后一层线性层的权重，前面巨大的主干网络完全不动

Full Fine-tuning 全量微调（放开全部参数）
预训练 Encoder不冻结，里面所有参数全部放开，参与梯度更新，预训练是给模型一个优质初始起点。

**只计算被掩码遮盖patch的损失**，可见的25%原始patch不参与损失计算

# 3、Contrastive representation learning

<img width="957" height="472" alt="Image" src="https://github.com/user-attachments/assets/676d7163-e8dd-4205-aa06-7d0f587e75b2" />


<img width="862" height="426" alt="Image" src="https://github.com/user-attachments/assets/f14c2803-af41-4ea6-9191-b6566ede4f8c" />

<img width="965" height="487" alt="Image" src="https://github.com/user-attachments/assets/9ee882d3-2d7d-455a-91ae-d2308769224d" />

图示这种损失函数的计算方法与softmax类似
 
 ### （1）SimCLR

<img width="941" height="427" alt="Image" src="https://github.com/user-attachments/assets/9aab9934-05e5-4cd0-873a-a2bf8de09bf5" />

1. $x$：原始输入图像
2. $\tilde{x}_i,\tilde{x}_j$：原图经过两种不同随机数据增强变换 $t,t' \sim \mathcal{T}$ 得到的两张增强视图（一对正样本）
3. $f(\cdot)$：主干编码器（CNN/ViT），输出图像基础表征 $h_i, h_j$，下游任务最终只用这个 $f$
4. $g(\cdot)$：投影头（Projection Network，多层全连接），把基础表征映射到对比学习专用空间，得到 $z_i,z_j$
5. $z_i,z_j$：投影后的特征向量，用来计算相似度损失
6. Maximize agreement：最大化 $z_i$ 和 $z_j$ 的相似度（拉近正样本）

核心：对同一张原图做两次独立随机增强，得到两张视觉有差异、但语义完全相同的图片，作为一对**正样本**；批次里其他所有图片的增强视图，全部作为**负样本**。