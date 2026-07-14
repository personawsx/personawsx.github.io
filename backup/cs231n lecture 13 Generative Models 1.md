# Supervised vs Unsupervised Learning 监督学习 vs 无监督学习
### Supervised Learning 监督学习
1. Data: $(x, y)$
   $x$ is data, $y$ is label
2. Goal: Learn a function to map $x \rightarrow y$
3. Examples: Classification, regression, object detection, semantic segmentation, image captioning, etc.

### Unsupervised Learning 无监督学习
1. Data: $x$
   Just data, no labels!
2. Goal: Learn hidden structure in data
3. Examples: Clustering（聚类）, dimensionality reduction（降维）, density estimation（密度估计）, etc.

# Generative vs Discriminative Models

## （1）Discriminative Model（判别模型）:
Learn a probability distribution $p(y|x)$

<img width="607" height="417" alt="Image" src="https://github.com/user-attachments/assets/a17f03a6-239c-4a4f-8c04-49bd6b8e1f2d" />

Problem：无法处理有问题的输入，依旧会给出所有可能的概率分布（即使输入不属于所有标签中的任意一个）

## （2）Generative Model:
Learn a probability distribution $p(x)$

<img width="602" height="452" alt="Image" src="https://github.com/user-attachments/assets/c1927147-7230-44d6-989c-d6b717b96615" />

目标：搞清楚 “什么样的图片是现实中合理、常见的”

## （3）Conditional Generative Model: Learn $p(x|y)$

<img width="623" height="428" alt="Image" src="https://github.com/user-attachments/assets/1485fc8a-e488-41dd-b0ec-82e27266ec4f" />

目标：学习每一类标签专属的图像分布

<img width="931" height="437" alt="Image" src="https://github.com/user-attachments/assets/c932f7df-943b-4645-8a83-d79545968c4f" />

一般来说，Generative model在实际中应用较少，Conditional generative models应用更加广泛

# Why Generative Models?

Modeling ambiguity: If there are many possible outputs x for an input y, we want to model $P(x | y)$

Language Modeling: Produce output text x from input text y

