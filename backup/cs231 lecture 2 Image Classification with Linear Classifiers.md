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

