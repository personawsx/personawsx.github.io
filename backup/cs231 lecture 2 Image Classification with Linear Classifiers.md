# 1、Data-driven Approaches
Problems：Semantic Gap 语义鸿沟（计算机和人之间沟通的脱节）
Challenges：
（1）Viewpoint Variation
（2）Illumination（光照）
（3）Background Clutter（背景凌乱）
（4）Occlusion（阻塞）
（5）Deformation（变形）
（6）Intraclass Variation（类内变异）
（7）Context（背景）

### Machine Learning
 a、Collect a dataset of images and labels
 b、Use Machine Learning algorithms to train a classifier
 c、Evaluate the classifier on new images

# 2、Nearest Neighbor Classifier
### (1)Memorize all data and labels:
```python
def train(images,labels):
# Machine learning!
  return model
```
### (2)Predict the label of the most similar training image:
```python
def predict(model,test_images):
#Use model to predict labels 
  return test_labels
```

### L1 distance：$$d_1(I_1,I_2) = \sum_{p} \left| I_1^p - I_2^p \right|$$

### test image
| 56 | 32 | 10  | 18  |
|----|----|-----|-----|
| 90 | 23 | 128 | 133 |
| 24 | 26 | 178 | 200 |
| 2  | 0  | 255 | 220 |
### training image
| 10 | 20 | 24  | 17  |
|----|----|-----|-----|
| 8  | 10 | 89  | 100 |
| 12 | 16 | 178 | 170 |
| 4  | 32 | 233 | 112 |
### pixel-wise absolute value differences（逐像素绝对值差）
| 46 | 12 | 14 | 1  |
|----|----|----|----|
| 82 | 13 | 39 | 33 |
| 12 | 10 | 0  | 30 |
| 2  | 32 | 22 | 108 |

所有像素差值求和：**add → 456**
python代码实现：
```python
import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

        return Ypred
```
上述最近邻算法其实是取k-近邻中k=1时的特殊情况，为了增强鲁棒性，有时会取k≥2，但此时会增加算法的复杂程度。
# 3、Setting Hyperparameters
## （1）k的选取：
一般来说，k越小，模型越简单，推理越快，能保留局部细节适合对小样本的分析；
k越大，抗干扰能力越强，但计算会变慢，局部细节被抹平。
## （2）L1、L2距离的选取：
#### L1 (Manhattan) distance/曼哈顿距离：
$$d_1(I_1, I_2) = \sum_{p} \left| I_1^p - I_2^p \right|$$

#### L2 (Euclidean) distance/欧几里得距离：
$$d_2(I_1, I_2) = \sqrt{\sum_{p} \left( I_1^p - I_2^p \right)^2}$$
##（3）数据集的处理：
a:Choose hyperparameters that works best on the training data. BAD:k=1 always works perfectly.
b:Choose hyperparameters that works best on the test data. BAD:No idea how algorithm will perform on new data.
c:Split data into train,validation(验证集);choose hyperparameters on validation and evaluate on test.
d: Cross-Validation:Split data into folds,try each fold as validation and average the result.
# 4、Linear Classifier

假设输入图片为 32×32×3 像素数组，展平后得到一维向量 x，长度：
32 × 32 × 3 = 3072

线性预测函数：
f(x, W) = Wx+b

维度匹配：
权重矩阵 W：10×3072
图像向量 x：3072×1
输出类别得分向量 f(x, W)：10×1
偏置项b:10×1，与输入无关

输出结果为 10 个数值，对应各类别的预测得分（class scores），其中 W 是模型可学习权重参数（parameters / weights）。

### How to choose W？
1、Define a loss function that quantifies our unhappiness with the scores across the training data.
2、Come up with a way of efficiently finding the parameters that minimize the loss function. (optimization)
$$
L = \frac{1}{N}\sum_{i} L_i\big(f(x_i, W),\,y_i\big)
$$

- $L$：整个训练数据集的平均损失，模型的优化目标，需要最小化该值
- $N$：训练集样本的总数量
- $i$：样本索引，代表第$i$个图像样本
- $L_i$：第$i$个样本的单样本损失函数，计算单张图片的预测误差
- $f(x_i,W)$：线性分类函数，输出第$i$张图像的类别得分向量
- $x_i$：第$i$张图片展平之后的一维特征向量
- $W$：模型的权重矩阵，为待学习的参数
- $y_i$：第$i$张图像的真实标签，是一个整数，用来标记正确类别

### 1. 线性得分输出
$$s = f(x_i;W)$$
- $x_i$：第$i$张输入图像向量
- $W$：权重矩阵（待学习参数）
- $s$：原始类别得分，也叫logits（未归一化的对数概率）

### 2. Softmax函数，将得分转为概率
$$P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_{j}e^{s_j}}$$
1. $e^{s_k}$：对得分做指数运算，保证数值大于0
2. $\sum_{j}e^{s_j}$：全部类别指数求和，实现归一化，让所有类别概率之和等于1

### 3. 单样本交叉熵损失
$$L_i = -\log P(Y=y_i|X=x_i)$$
- $y_i$：第$i$张图片的真实标签
- $P(Y=y_i|X=x_i)$：模型分配给真实类别的预测概率
- 负对数：真实类别预测概率越高，损失值越小

### 4. 交叉熵信息论解释
$$H(P,Q)=H(P)+D_{KL}(P||Q)$$
- $P$：真实标签的独热分布
- $Q$：模型预测的概率分布
- $H(P)$：真实分布熵（常量）
- $D_{KL}(P||Q)$：KL散度，衡量预测分布与真实分布之间的差距，优化本质就是最小化KL散度