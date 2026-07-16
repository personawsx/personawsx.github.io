<img width="913" height="477" alt="Image" src="https://github.com/user-attachments/assets/394d504d-7594-49ef-89e4-7b2b4f4581d8" />

### （1）显式表示（Explicit）：
点云（Point cloud）
多边形网格（Polygon mesh）
细分曲面、非均匀有理B样条（Subdivision, NURBS）
 ……

### （2）隐式表示（Implicit）：
水平集（Level sets，图内拼写笔误Lever）
代数曲面（Algebraic surface）
距离函数（Distance functions）
 ……

每种表示方案都最适配特定任务、特定类型的几何体

<img width="692" height="407" alt="Image" src="https://github.com/user-attachments/assets/1f3cf86d-b3a1-4840-aa29-c79ae463cce0" />

### 几何表示方案的考量因素
1. **存储需求**：数据需要能在计算机中保存
2. **模型创建**：生成全新模型
    输入交互方式、操作界面……
3. **几何运算**
    编辑、简化、平滑、滤波、修复……
4. **渲染输出**
    光栅化、光线追踪、神经渲染……
5. **动画制作**
 
<img width="913" height="523" alt="Image" src="https://github.com/user-attachments/assets/e4bfdde3-3ab5-47d8-b6f3-8a2a017efba9" />

# 一、Explicit

### 1、Point Clouds

Simplest representation: only points, no connectivity
Collection of $(x,y,z)$ coordinates, possibly with normal
Points with orientation are called surfels（面元）
Often results from scanners
Potentially noisy
Registration of multiple images

### 优点：
Easily represent any kind of geometry
Useful for large datasets
### Other limitations:
Difficult to draw in undersampled regions（采样稀疏处难以绘制）
No simplification（模型简化） or subdivision（细分曲面）
No direction smooth rendering（基于方向的平滑渲染）
No topological information（缺少拓普信息）

<img width="537" height="117" alt="Image" src="https://github.com/user-attachments/assets/56d55dea-a52c-493e-9bc4-dabd2905e2e8" />

### 2、Polygon meshes（多边形网格）

<img width="622" height="170" alt="Image" src="https://github.com/user-attachments/assets/2775798b-28ea-4f5b-911e-659ed859157c" />

Subdivision（细分曲面）
需要精细：增加网格数量；需要快速：减少网格数量

Mesh Regularization：用三角形网格连接

### 3、Shape Representations：
Non-parametric：Points，Meshes
Parametric Representations：

<img width="290" height="177" alt="Image" src="https://github.com/user-attachments/assets/7ab4c1a1-eeaf-4dbe-a2fe-29aa99001d08" />

<img width="742" height="498" alt="Image" src="https://github.com/user-attachments/assets/50ca378d-0119-4272-b30f-439e00a534c3" />

上述方法都属于显性采样，所有点都被直接给出来。
优点：便于采样。
缺点：很难判断任意一点是在物体的内部还是外部。

# 二、Implicit

<img width="820" height="497" alt="Image" src="https://github.com/user-attachments/assets/a3778d6a-0171-4eb4-8012-e63bf1c4d4d1" />

优点：
可以轻松判断点是在物体的外部还是内部。
对于形状复杂的物体，可以通过函数的组合来实现表示。
缺点：对于 $f(x,y,z)=0$，难以直接采样。 

### 1、Level Set Methods

<img width="788" height="500" alt="Image" src="https://github.com/user-attachments/assets/145ce71e-f79a-4462-a27b-d7edc6b8e6af" />

### 2、AI + Geometry: Tasks

#### P(S) or P(S|c) --- Generative models
Learning (conditional) shape priors
Shape generation, completion, & geometry data processing

#### P(c|S) --- Discriminative models
Learning shape descriptors
Shape classification, segmentation, view estimation, etc.

相关变量含义：
## 1. $S$ = Shape，三维几何形状
## 2. $c$ = conditioning / condition，上下文标签

#### Joint modeling of 3D and 2D data
Large-scale 2D datasets & very good pretrained models
Differentiable projection/back-projection & differentiable/neural rendering

#### Joint modeling of multi-modal data beyond visual (e.g., text)

### 3、如何处理3D数据？

#### （1）

<img width="872" height="392" alt="Image" src="https://github.com/user-attachments/assets/b38c1a38-afd4-4837-b16c-353306361b74" />

<img width="715" height="338" alt="Image" src="https://github.com/user-attachments/assets/c0a4997c-812a-4850-86c6-7019189218c3" />

#### （2）

<img width="872" height="485" alt="Image" src="https://github.com/user-attachments/assets/0e9d46bc-8991-4627-8385-8ec726f2e9c9" />

#### （3）3D voxels（三维体素）

<img width="885" height="302" alt="Image" src="https://github.com/user-attachments/assets/d5091126-aba5-467a-bbc0-065b5dfcf3ca" />

改进：将GAN生成的三维形状进行投影，然后利用循环网络将深度图转换为彩色图像

<img width="891" height="322" alt="Image" src="https://github.com/user-attachments/assets/fa158ea0-9a05-4298-b3eb-584adba9b37a" />

3D voxels把三维空间均匀切分成无数大小完全相同的小立方体，类似 2D 图像像素。
特点：均匀网格，空间所有区域分辨率一致；
缺点：空的空白区域也要存完整体素，稀疏物体（椅子、兔子）显存 / 存储爆炸，高分辨率几乎不可用。

Solution：
Octree 八叉树
是体素的分层压缩存储结构，专门解决均匀体素冗余问题。
三维立方体空间每次递归切分为 8 个子立方体（上下、左右、前后），因此叫八叉树。

#### （4）Octave Tree Representations（八叉树）