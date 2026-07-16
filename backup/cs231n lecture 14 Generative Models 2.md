# 1、Generative Adversarial Networks
Generative Adversarial Networks give up on modeling p(x) ，but allow us to draw samples from p(x) .

<img width="978" height="501" alt="Image" src="https://github.com/user-attachments/assets/584a4ce3-9ff0-4a9f-89bf-ed87659cdc07" />

相关符号：
1. $z \sim p(z)$：从标准正态分布采样随机隐向量；
2. $G(z)$：生成器，输入 $z$输出伪造样本；
3. $p_G$：生成器造出图片的分布；
4. $p_{\text{data}}$：数据集真实图片的分布；
5. $D(x)$：判别器，输入图片输出“是真实图的概率”。

对抗思路：
1. $p_{\text{data}}(x)$：真实数据分布
我们手里的真实图片、真实样本，全部来自这个真实世界分布 $p_{\text{data}}(x)$；
模型最终目标：能生成和真实数据一模一样的样本，也就是让模型生成分布等于真实分布 $p_G=p_{\text{data}}$。
2. 核心Idea
引入隐变量 $z$，先验 $p(z)$ 一般取标准正态分布（和VAE的 $z$先验一样）；
随机采样一个 $z$，送入**生成器G**，输出假图片 $x=G(z)$；
生成器产出的图片服从生成分布 $p_G$，我们希望训练到 $p_G$ 和真实分布完全重合。
将训练得到的伪造图和真实图一起送入判别器网络 $D$。
判别器输出二分类结果：`Fake` / `Real`。判别器越难以识别真假，说明模型效果越好

<img width="966" height="301" alt="Image" src="https://github.com/user-attachments/assets/c6442ddf-e20c-4bd5-814a-faad97453950" />

**用极小极大博弈（minimax game），同时对抗训练生成器G、判别器D**。
### 公式：

$$
\min_G \max_D \left( \mathbb{E}_{x\sim p_{\text{data}}} \big[\log D(x)\big] + \mathbb{E}_{z\sim p(z)} \big[\log\big(1-D(G(z))\big)\big] \right)
$$

相关符号定义：

### 第一项： $\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]$ 真实图片损失项
1. $x\sim p_{\text{data}}$：从真实数据集里采样真实图片$x$；
2. $D(x)$：判别器输出，代表这张图是「真实图片」的概率，取值0（假）或1（真）；
3. $\log D(x)$：对数损失；
4. $\mathbb{E}$：期望，对全部真实图片取平均。

### 第二项： $\mathbb{E}_{z\sim p(z)}\big[\log\big(1-D(G(z))\big)\big]$ 假图片损失项
1. $z\sim p(z)$：从标准正态先验随机采样隐向量 $z$；
2. $G(z)$：生成器接收 $z$，输出一张伪造图片；
3. $D(G(z))$：判别器给假图打分，代表它认为这张假图是真实图的概率；
4. $1-D(G(z))$：判别器认为这张是假图的概率；
5. $\log(1-D(G(z)))$：对数损失；
6. $\mathbb{E}$：对大量采样的 $z$取平均。

### 外层符号 $\min_G \max_D$：极小极大博弈
### $\max_D$：先固定G，最大化整个式子（训练判别器D），即在训练D时，生成器G权重冻结不动，只更新D。
### $\min_G$：再固定D，最小化整个式子（训练生成器G）

$\max_D$的目标是把括号里这一整项尽可能拉大：
1. 真实图片输入： $D(x)=1$ → $\log D(x)=\log1=0$，值最大；
2. G生成的假图输入： $D(G(z))=0$ → $\log(1-D(G(z)))=\log1=0$，值最大；
即给真实图打1、假图打0，最大化目标值。

 $\min_G$的目标是把括号里这一整项尽可能缩小：
此时固定判别器D，只更新生成器G，生成器只关心伪造图片对应的第二项，第一项来自真实图片、不受G控制，G只能改变假图得分。
1. G生成的假图输入： $D(G(z))=1$ → $\log(1-D(G(z)))=\log0=-\infty$，该项会变得极小；
2. 真实图片项 $\log D(x)$ 和生成器无关，无法调整；
即生成器努力让假图被判别器当成真实图、给假图打1，以此压低整体目标函数数值。

<img width="972" height="491" alt="Image" src="https://github.com/user-attachments/assets/7b788998-d675-4a33-87cb-c492ae5e79ac" />

+对应最大化，采用梯度上升；-号对应最小化，采用梯度下降

<img width="948" height="491" alt="Image" src="https://github.com/user-attachments/assets/2e054c9f-b6c9-4df4-bcf7-20f541db8f39" />

Problem：对于蓝色曲线 $\log(1-D(G(z)))$，左端（ $D≈0$）平缓、梯度微弱，意味着反向传播时几乎没有更新信号，生成器停滞、无法优化。
Solution：不再最小化 $\log(1-D(G(z)))$，改为**最小化 $-\log(D(G(z)))$**。

<img width="967" height="501" alt="Image" src="https://github.com/user-attachments/assets/262fe80a-ee46-445a-95b5-b62e9d2201af" />

为何GAN极小极大目标函数设计合理？
内层 $\max_D$：固定生成器时，理论最优判别器 $D_G^*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_G(x)}$；
外层 $\min_G$：在最优判别器基础上，最优生成结果是生成分布等于真实分布 $p_G=p_{\text{data}}$；
两点现实局限：神经网络拟合能力有限；有限数据下无法保证训练收敛。

<img width="977" height="462" alt="Image" src="https://github.com/user-attachments/assets/0b4b681c-392a-42a4-baf8-79f6fa6dca3f" />

<img width="968" height="457" alt="Image" src="https://github.com/user-attachments/assets/702da444-e03e-4564-bd8d-f8abbe9fe055" />

### VAE 与 GAN 核心差异简要对比
VAE：编码器(Encoder)+解码器(Decoder)，**双向映射**
   $x \Rightarrow z$（编码）、 $z \Rightarrow x$（解码）
GAN：生成器G+判别器D，**仅单向映射**
  只有 $z \Rightarrow x$，无编码器，无法从图片反推隐变量z

VAE：有规整连续隐空间，支持图像重建、插值、原图编辑
 GAN：隐空间无显式反向映射，不能直接编码真实图片

VAE：图像偏模糊，稳定易训练
 GAN：图像清晰度更高，易出现梯度消失、模式崩溃，训练不稳定

# 2、Diffusion Models：Intuition

GAN通过生成器学习确定性映射直接将z映射到x，扩散模型采用更加间接的方式

<img width="963" height="505" alt="Image" src="https://github.com/user-attachments/assets/109b19c1-3de6-4f1c-ac7a-30533a00233b" />

### 扩散模型核心概括
1. **前向扩散（训练）**：原图 $x_0$ 随时间步 $t$ 逐步叠加高斯噪声，最终变成纯噪声图；
2. **网络训练目标**：训练网络 $f_\theta(x_t,t)$，输入带噪图与时间步，学习单次小幅去除噪声；
3. **反向生成（推理）**：从随机纯噪声出发，反复调用去噪网络迭代去噪，逐步还原清晰原图；

<img width="952" height="463" alt="Image" src="https://github.com/user-attachments/assets/182ccc0c-2d5c-4976-aa0a-77e304abf891" />

训练阶段：
存在两种分布，真实数据分布 $p_{\text{data}}$（原图区域）、噪声分布 $p_{\text{noise}}$（高斯噪声区域）
每一步同时采样三样东西：真实图 $x$、高斯噪声 $z$、0~1均匀时间步 $t$。
中间样本： $x_t=(1-t)x + tz$
目标速度向量： $v=z-x$
网络 $f_\theta(x_t,t)$ 输入中间图+时间步，预测速度 $v$；损失为预测速度与真实速度的L2距离。
代码实现如下：

<img width="975" height="456" alt="Image" src="https://github.com/user-attachments/assets/4bb44095-a57e-49d1-ac8a-2a8ad927f488" />

<img width="977" height="457" alt="Image" src="https://github.com/user-attachments/assets/997d225d-b374-4179-a633-5e0e63fc9914" />

采样阶段是从z还原到x，时间步t从1降至0，即沿着反方向走。

上述都是无条件生成模型，我们真正关心地是条件生成模型，如下：

<img width="980" height="487" alt="Image" src="https://github.com/user-attachments/assets/9b8c1478-66a2-4383-a87a-61693bdd9c1e" />

Problem：Can we control how much we “emphasize” the conditioning y?
一般来说，将y喂进模型之后，只有 “用 / 不用” 两种状态；但实际生成时我们想要可调强度：
弱强调：轻微贴合y，画面自由度高，和描述略有出入；
强强调：严格贴合y，画面完全匹配输入条件，不能跑偏。
Solution：Classifier-Free Guidance（CFG）

<img width="963" height="500" alt="Image" src="https://github.com/user-attachments/assets/952cafad-1bc6-4d33-9f2a-a76403509e2a" />

关键：每次训练有50%概率把有效条件 $y$ 替换为空条件 $y_\emptyset$，即模型可以同时具备条件生成和无条件生成两种能力

### 三个速度向量几何含义
1. $v^\emptyset$：空条件预测速度，往全部真实图片区域走，不区分类别；
2. $v^y$：带条件y的速度，专门往匹配y的图片子集走；
3. $v^{cfg}$：CFG加权融合后的最终速度，比单纯 $v^y$更偏向符合y的区域，即：

$$
v^{cfg} = (1+w)v^y - wv^\emptyset
$$

其中 $w$ = CFG权重（引导强度，人为可调）
$w$越大， $v^{cfg}$越偏向 $p(x|y)$，生成图像严格匹配提示词；
$w=0$ 等价普通条件生成，无强化；
$w$过小，画面会偏离输入的描述y。

Problem：原生均匀t采样不合理，因为中等噪声最难，但训练样本较少
Solution：Logit-Normal 非均匀采样，提升中间噪声 t 的采样概率

Problem：扩散模型在高分辨率情况下工作状态不理想，因为高分辨率需要更加精细的纹理信息、光影细节，高噪声容易损失掉这部分信息，如果t按均匀采样或者在中间采样，会导致高噪声情况下采样少，难以还原

# 3、Latent Diffusion Models（LDM 潜扩散模型）

<img width="977" height="505" alt="Image" src="https://github.com/user-attachments/assets/b7615828-e880-4e16-9713-20e23a615a28" />

<img width="977" height="501" alt="Image" src="https://github.com/user-attachments/assets/1fdb99b0-2983-4a39-9927-53e6a78989ea" />

<img width="986" height="495" alt="Image" src="https://github.com/user-attachments/assets/601d5595-3676-4637-9c31-adb445752fa6" />

Problem：原始扩散模型计算量大，内存开销大
Solution：先引入VAE，原图 $H×W×3$ 经过Encoder下采样D倍，得到小潜变量 $H/D × W/D × C$；Decoder可以把latent还原回高清原图

Problem：单纯标准VAE解码出来的图像普遍模糊
Solution：VAE训练时额外加一个**判别器Discriminator**，对抗训练提升细节清晰度

# 4、DiT Diffusion Transformer（基于 Transformer 的扩散骨干）

<img width="855" height="480" alt="Image" src="https://github.com/user-attachments/assets/343db7f2-9e4f-47cb-8e7e-6fccc0bb4718" />



# 5、Text-to-Image

<img width="977" height="505" alt="Image" src="https://github.com/user-attachments/assets/23c6b506-08f4-4699-8ef5-1b372b6cbed3" />

# 6、Text-to-Video

<img width="987" height="490" alt="Image" src="https://github.com/user-attachments/assets/c06d1622-c81f-44e0-b910-5b299136c412" />

# 7、Diffusion Distillation 扩散蒸馏（解决采样慢的痛点）

<img width="982" height="505" alt="Image" src="https://github.com/user-attachments/assets/3a115995-1007-489c-98f8-32e0c15d3af6" />

# 8、Generalized Diffusion

<img width="882" height="298" alt="Image" src="https://github.com/user-attachments/assets/b2348494-4891-4f2c-b935-faf60f5595a2" />

实际中，一般以各种函数来替代上文所讲的模型中的相关变量：

$$
\begin{aligned}
a(t) &= 1 - t \\
b(t) &= t \\
c(t) &= -1 \\
d(t) &= 1
\end{aligned}
$$

How do we choose these functions？
Usually through some mathematical formalism.

（1）

<img width="852" height="356" alt="Image" src="https://github.com/user-attachments/assets/efd7224e-2e33-45b2-9519-2959ca614754" /> 

（2）

<img width="826" height="333" alt="Image" src="https://github.com/user-attachments/assets/a79b2558-27c6-48fe-8a6a-f8e7854c4c5d" />

（3）

<img width="598" height="340" alt="Image" src="https://github.com/user-attachments/assets/8e4acfb8-a973-4919-8652-3ff98083cf04" />