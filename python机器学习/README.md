请安装插件后阅读公式：

https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima

scikit-learn官网：

https://scikit-learn.org/


- [1.数据预处理](#1数据预处理)
  - [1-1.加载数据集](#1-1加载数据集)
  - [1-2.切分数据集](#1-2切分数据集)
  - [1-3.归一化](#1-3归一化)
  - [1-4.sklearn接口](#1-4sklearn接口)
- [2.K近邻算法](#2k近邻算法)
  - [2-1.简介](#2-1简介)
  - [2-2.评价标准：](#2-2评价标准)
  - [2-3.python实现](#2-3python实现)
  - [2-4.sklearn接口](#2-4sklearn接口)
  - [2-5.knn中的超参数](#2-5knn中的超参数)
  - [2-6.对knn做超参数网格搜索](#2-6对knn做超参数网格搜索)
  - [2-7.knn完成回归任务](#2-7knn完成回归任务)
- [3.线性回归](#3线性回归)
  - [3-1.评价标准：](#3-1评价标准)
  - [3-2.简单线性回归](#3-2简单线性回归)
  - [3-3.多元线性回归](#3-3多元线性回归)
  - [3-4.python实现](#3-4python实现)
    - [3-4-1.简单线性回归的python实现](#3-4-1简单线性回归的python实现)
    - [3-4-2.多元线性回归的python实现](#3-4-2多元线性回归的python实现)
  - [3-5.sklearn接口](#3-5sklearn接口)
  - [3-6.线性回归的可解释性](#3-6线性回归的可解释性)
- [4.梯度下降](#4梯度下降)
  - [4-1.批量梯度下降](#4-1批量梯度下降)
  - [4-2.随机梯度下降](#4-2随机梯度下降)
  - [4-3.小批量梯度下降](#4-3小批量梯度下降)
  - [4-4.python实现](#4-4python实现)
    - [4-4-1.模拟梯度下降](#4-4-1模拟梯度下降)
    - [4-4-2.批量梯度下降](#4-4-2批量梯度下降)
    - [4-4-3.随机梯度下降](#4-4-3随机梯度下降)
    - [4-4-4.小批量梯度下降](#4-4-4小批量梯度下降)
  - [4-5.sklearn接口](#4-5sklearn接口)
  - [4-6.梯度验证](#4-6梯度验证)
- [5.主成分分析PCA](#5主成分分析pca)
- [6.多项式回归和模型泛化](#6多项式回归和模型泛化)
  - [6-1.python实现](#6-1python实现)
  - [6-2.sklearn接口](#6-2sklearn接口)
  - [6-3.pipeline构造多项式回归函数](#6-3pipeline构造多项式回归函数)
  - [6-4.过拟合和欠拟合](#6-4过拟合和欠拟合)
  - [6-5.解决过拟合](#6-5解决过拟合)
  - [6-6.交叉验证](#6-6交叉验证)
  - [6-7.正则化](#6-7正则化)
    - [6-7-1.岭回归](#6-7-1岭回归)
    - [6-7-2.LASSO回归](#6-7-2lasso回归)
    - [6-7-3.Ridge 和 Lasso 比较](#6-7-3ridge-和-lasso-比较)
    - [6-7-4.L1 L2正则](#6-7-4l1-l2正则)
    - [6-7-5.弹性网](#6-7-5弹性网)
- [7.逻辑回归](#7逻辑回归)
  - [7-1.介绍](#7-1介绍)
  - [7-2.python实现](#7-2python实现)
  - [7-3.决策边界](#7-3决策边界)
  - [7-4.sklearn接口](#7-4sklearn接口)
  - [7-5.二分类改多分类](#7-5二分类改多分类)
    - [7-5-1.OVR ：one vs rest](#7-5-1ovr-one-vs-rest)
    - [7-5-2.OVO：one vs one](#7-5-2ovoone-vs-one)
- [8.分类任务评价标准](#8分类任务评价标准)
  - [8-1.混淆矩阵，精确率，召回率](#8-1混淆矩阵精确率召回率)
    - [8-1-1.python实现](#8-1-1python实现)
    - [8-1-2.sklearn接口](#8-1-2sklearn接口)
    - [8-1-3.多分类的混淆矩阵](#8-1-3多分类的混淆矩阵)
  - [8-2.F1 Score](#8-2f1-score)
  - [8-3.PR曲线](#8-3pr曲线)
  - [8-4.ROC曲线](#8-4roc曲线)
  - [8-5.AUC](#8-5auc)
- [9.支撑向量机SVM](#9支撑向量机svm)
  - [9-1.简介](#9-1简介)
  - [9-2.最优化](#9-2最优化)
  - [9-3.软间隔](#9-3软间隔)
  - [9-4.sklearn接口](#9-4sklearn接口)
  - [9-5.多项式特征的svm](#9-5多项式特征的svm)
  - [9-6.核函数](#9-6核函数)
  - [9-7.解决回归问题](#9-7解决回归问题)
- [10.决策树](#10决策树)
  - [10-1.信息熵](#10-1信息熵)
  - [10-2.基尼系数](#10-2基尼系数)
  - [10-4.sklearn接口](#10-4sklearn接口)
  - [10-5.解决回归问题](#10-5解决回归问题)
  - [10-6.局限性](#10-6局限性)
- [11.集成学习和随机森林](#11集成学习和随机森林)
  - [11-1.简介](#11-1简介)
  - [11-2.软投票](#11-2软投票)
  - [11-3.bagging和pasting](#11-3bagging和pasting)
  - [11-4.sklearn接口](#11-4sklearn接口)
  - [11-5.随机森林和Extra-Trees](#11-5随机森林和extra-trees)
  - [11-7.提升方法](#11-7提升方法)
    - [11-7-1.ada boosting](#11-7-1ada-boosting)
    - [11-7-2.gradient boosting](#11-7-2gradient-boosting)
    - [11-7-3.stacking](#11-7-3stacking)
  - [11-8.总结](#11-8总结)

![image-20201201140737173](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201201140737173.png)

**机器学习分类一：**

监督学习--数据有标记：分类 回归

非监督学习--数据无标注：聚类  降维

半监督学习--部分数据有标记部分没有：通常用无监督学习做处理，然后监督学习训练和预测

增强学习：模型在具体环境中，不断用当前环境的数据来更新模型



**机器学习分类二：**

批量学习：在数据集上学习一个模型，投入使用环境中，模型不再改变，或一段时间后再批量学习

在线学习：得到模型后，每输入一个样例，做出预测，并根据真实结果(如一分钟后的股价)更新模型



**机器学习分类三：**

参数学习：把问题预设成一个数学模型  从数据中学习数学模型的参数

非参数学习：不把问题预设成一个数学模型  没有数学模型中的参数  不代表没有要学习的参数 

**超参数和模型参数**
超参数：模型训练前要指定的参数(调参)
模型参数：模型要学习更新的参数


## 1.数据预处理
### 1-1.加载数据集

```python
# 从外部导入数据集 假设导入后是列表的形式
# 原始数据
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 转换成numpy之后再进行后续操作
data_X = np.array(raw_data_X)
data_y = np.array(raw_data_y)
```

### 1-2.切分数据集

```python
import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    if seed:
        np.random.seed(seed)
	
    # 得到一个打乱的序号 
    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]
	
    # 按照打乱的序号分训练集和测试集
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
```

### 1-3.归一化

最值归一化 （x-xmin）/(xmax-xmin) 受异常点影响大 适用于有明显的取值边界 如像素0-255 成绩0-100 
均值方差归一化 (x-mean)/std 受异常点影响小 适用于没有明显取值边界的数据 如收入
对测试集的归一化：用训练集的统计值(真实的测试集是拿不到的 而且归一化其实是训练模型算法的一部分)

```python
# 均值方差归一化
import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
	
    # 传入训练集 分别求它各个维度特征的均值和方差
    def fit(self, X):
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])
        return self
	
    # 传入要归一化的数据X 根据训练集求得的均值和方差给X做归一化
    def transform(self, X):
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]
        return resX
```

### 1-4.sklearn接口

```python
# 导入数据
from sklearn import datasets
iris = datasets.load_iris() # 得到一个字典
X = iris.data
y = iris.target  # 此时加载进来的数据已经是numpy.ndarray类型了

# 切分数据集
from skleran.modle_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# 归一化
from sklearn.preprocessing import StandardScaler
Scalar = StandardScaler()
standardScalar.fit(X_train)  # 求训练集各个维度特征的均值和方差 
X_train = Scalar.transform(X_train)  # 归一化后的训练集
```

## 2.K近邻算法

![image-20201201172225829](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201201172225829.png)

### 2-1.简介

**原理：**

一个样本，求距离它最近的k个样本所属的类别，哪个类别占比大，就认为这个样本是哪个类别

**特点：**

knn本身不需要训练参数 训练集本身就是模型 输入样例根据数据集得出预测结果

knn可以解决多分类 也可以解决回归问题:回归值等于最近的k个点的回归值的平均数

**缺点：**

效率低  训练集(m,n) 每一个样本的时间复杂度o(m*n)

错误值对算法影响特别大，比如最近的三个样本中有两个错误的，那么预测出来大概率也是错误的

knn没有解释性  人不能从模型中学到信息

### 2-2.评价标准：

准确率accuracy = 预测正确的样本数 / 预测样本总数

```python
# python实现
# y_test和y_predict都是一个向量 每一个元素是一个样本x
acc = np.sum(y_predict == y_test) / len(y_test)

# sklearn接口
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)
```

### 2-3.python实现

```python
# 简单实现
import numpy as np
from math import sqrt
from collections import Counter

# 原始数据
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 转换成numpy
data_X = np.array(raw_data_X)
data_y = np.array(raw_data_y)

x = np.array([8.093607318, 3.365731514]) # 待预测样本

distance = [sqrt(np.sum((point - x)**2)) for point in data_X] # 求与各点的距离

nearest = np.argsort(distance) # 距离从小到大排序 返回序号

k = 6
top_k = data_y[nearest[:k]] # 按照排序的序号取出前k个的标签
 
votes = Counter(top_k) # 统计各个类别出现的次数

predict_y = votes.most_common(1)[0][0] # 取出出现次数最多的类别 
```

按照sklearn的格式封装成类

```python
class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None
	
    # 拟合数据集(训练) ： knn训练过程就是保存数据集
    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self
	
    # 传入要预测的样本矩阵 返回预测结果的nparray形式
    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
	
    # 传入要预测的单个样本 返回预测结果
    def _predict(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    # 计算准确率
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return np.sum(y_predict == y_test) / len(y_test)
	
    # print实例时 打印下面的类介绍
    def __repr__(self):
        return "KNN(k=%d)" % self.k
```

### 2-4.sklearn接口

文档链接：

https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

```python
"""
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
"""

# 示例
import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()  # sklearn内置的手写数字数据集 返回一个字典
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)  # 切分数据集

knn_clf = KNeighborsClassifier(n_neighbors=3)  # 实例化
knn_clf.fit(X_train, y_train)  # 拟合
y_predict = knn_clf.predict(X_test)  # 预测
score = knn_clf.score(X_test, y_test)  # 评分 默认使用accuracy 注意这里输入的是测试集的xy不是预测的y和真实y
```

### 2-5.knn中的超参数

**1.k**  代表选取最近的多少个样本作为参考

**2.weight**   如图绿色样本最近的三个点中有两个蓝色 此时knn判断绿色样本属于蓝色类别 但实际上它离红色类别更近 因此knn除了考虑最近k个样本的类别以外 还应该考虑他们的距离 越近的权重应当越大 一般权重取距离的倒数 **使用距离时要做数据的标准化**

![image-20201201180240215](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201201180240215.png)

**3.明可夫斯基距离参数p**、

在knn的实现中 我们使用的是**欧式距离**

a b 两个样本在各个维度的差的平方的和开根号

![image-20201201180709328](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201201180709328.png)

除此之外还可以使用**曼哈顿距离**

a b 两个样本各个维度的差的绝对值的和

![image-20201201180900764](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201201180900764.png)

其实欧氏距离是两点的连线距离  曼哈顿距离是两点的折现距离

绿色是欧氏距离  红紫黄都是曼哈顿距离 他们都相同

![image-20201201181005307](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201201181005307.png)

上述两种距离其实都是取不同p值的明可夫斯基距离

![image-20201201181229486](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201201181229486.png)

### 2-6.对knn做超参数网格搜索

k = 1,2,3,4,5,6,7,8,9,10

weights = uniform (只数个数 不加权)  distance(计算距离 距离的倒数做加权)

p = 1,2,3,4,5  只有计算距离时才使用这个超参数

超参数网格搜索就是在上面所有的超参数取值中找到一组效果最好的超参数  像是在三维网格上找一个最好的点

```python
best_k = 0
best_weights = ''
best_p = 0
best_score = 0.0
for k in range(1,11):
    for weights in ['uniform','distance']:
        if weights == 'distance':
            for p in range(1,6):
                knn_cls = KNeighborsClassifier(k,weights=weights,p=p)
                knn_cls.fit(x,y)
                score = knn.score(test_x,test_y)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_weights = weights
                    best_p = p
        else:
            p=0
            knn_cls = KNeighborsClassifier(k,weights=weights,p=p)
            knn_cls.fit(x,y)
            score = knn.score(test_x,test_y)
            if score > best_score:
                best_score = score
                best_k = k
                best_weights = weights
                best_p = p
```

sklearn封装的接口

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1,6)]
    }
]

knn_cls = KNeighborsClassifier()  # 创建实例
grid_search = GridSearchCV(knn_cls, param_grid, n_jobs=-1, verbose=1)  # 创建搜索实例
grid_search.fit(X_train_standard, y_train)  # 拟合数据做搜索

# sklearn中所有内部计算出来且不输出的参数 最后都跟一个下划线
grid_search.best_params_  # 最好的参数
grid_search.best_score_  # 交叉验证下的最好成绩
grid_search.best_estimator_  # 最好的模型
grid_search.best_estimator_.score(X_test_standard, y_test)  # 最好模型的准确率
```

### 2-7.knn完成回归任务

![image-20201201190616918](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201201190616918.png)

把top_k的标签取出来后 标签不再是类别 所以不用统计类别数 而是把标签的数值相加求平均(或加权平均)

```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train_standard, y_train)
knn_reg.score(X_test_standard, y_test)
```

## 3.线性回归

**原理：**

假设数据的特征之间有线性关系，通过拟合数据集得到一组线性参数，使得标签值等于特征之间的线性组合，把预测样本的特征带入这组线性参数，得到预测的数值(回归任务结果是连续值)

**特点：**

解决回归问题   有很好的可解释性   是许多非线性模型的基础

**缺点：**

不能拟合非线性关系的数据

### 3-1.评价标准：

$$
\sum_{i=1}^{m}\left(y_{\text {test}}^{(i)}-\hat{y}_{\text {test}}^{(i)}\right)^{2}
$$

SE：square error 平方误差  能够表达预测值和真实值的差距有多大 但是受m的影响 即样本越多 误差越大  


$$
\frac{1}{m} \sum_{i=1}^{m}\left(y_{t e s t}^{(i)}-\hat{y}_{t e s t}^{(i)}\right)^{2}
$$

MSE：mean square error 均方误差  不受m影响 但是量纲难以解释 比如任务是拟合波士顿房价，y的量纲是万元，那么做了平方后mse=10就代表了每个样本的误差是10(万元的平方)  难以理解


$$
\sqrt{\frac{1}{m} \sum_{i=1}^{m}\left(y_{\text {test }}^{(i)}-\hat{y}_{\text {test }}^{(i)}\right)^{2}}
$$

RMSE：root mean square error 均方根误差   很好


$$
\frac{1}{m} \sum_{i=1}^{m}\left|y_{\text {test }}^{(i)}-\hat{y}_{\text {test }}^{(i)}\right|
$$

MAE：mean absolute mean  平均绝对误差  也可以用作线性回归任务的评价标准 

R2：r square 最好的回归任务评价标准  sklearn的线性回归默认使用R2的评价方式

![image-20201202103601916](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201202103601916.png)

R2最大可为1，此时所以预测结果均完全正确

R2等于0时，说明模型和基准模型效果一致

R2小于0时，说明模型效果不如基准模型，数据的特征可能并不存在线性关系 线性回归不适用

整理一下R2的公式
$$
R^{2}=1-\frac{\sum_{i}\left(\hat{y}^{(i)}-y^{(i)}\right)^{2}}{\sum_{i}\left(\bar{y}-y^{(i)}\right)^{2}}=1-\frac{\left(\sum_{i=1}^{m}\left(\hat{y}^{(i)}-y^{(i)}\right)^{2}\right) / m}{\left(\sum_{j=1}^{m}\left(y^{(i)}-\bar{y}\right)^{2}\right) / m}=1-\frac{M S E(\hat{y}, y)}{\operatorname{Var}(y)}
$$

代码实现

```python
# python实现
np.sum((y_predict-y_test)**2)  # SE
np.mean((y_predict-y_test)**2)  # MSE
np.sqrt(np.mean((y_predict-y_test)**2))  # RMSE
np.mean(np.abs(y_predict-y_test))  # MAE
1-(np.mean((y_predict-y_test)**2))/np.var(y_test)  # R2

# sklearn接口
from sklearn.metrics import mean_squared_error, mean_absolute_error, s2_score
mean_squared_error(y_predict, y_test)
```

### 3-2.简单线性回归

假设样本只有一个特征 那么最佳拟合方程可以假设为 y=ax+b 这种问题称为简单线性回归

![image-20201202094452512](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201202094452512.png)

表达差距的函数，其实就是损失函数，要最小化这个函数，就是最小化差距。

y - y_hat 不能表达差距 因为这个真实值减去预测值有正有负 m个样本的差值相加可能是0

|y - y_hat|可以表达差距 但是这个函数不是处处可导的  要最小化差距(求这个函数的极小值)  比较麻烦 **(注意这里和MAE一个是损失函数 一个是评价函数)**

(y - y_hat)**2 可以表达差距 同时也处处可导 方便求这个函数的极小值

**用最小二乘法来求损失函数的最小值**    (把y_hat=ax+b带入损失函数然后对ab求导并使其为0)
$$
a=\frac{\sum_{i=1}^{m}\left(x^{(i)}-\bar{x}\right)\left(y^{(i)}-\bar{y}\right)}{\sum_{i=1}^{m}\left(x^{(i)}-\bar{x}\right)^{2}}
$$

$$
b=\bar{y}-a \bar{x}
$$

a的分子和分母都可以看成是两个向量的点乘 -- 各元素相乘相加

### 3-3.多元线性回归

样本有多个特征的线性回归

假设第i个样本的特征
$$
X^{(i)}=\left(X_{0}^{(i)}, X_{1}^{(i)}, X_{2}^{(i)}, \ldots, X_{n}^{(i)}\right)
$$
要拟合的参数
$$
\theta=\left(\theta_{0}, \theta_{1}, \theta_{2}, \ldots, \theta_{n}\right)^{T}
$$
拟合后的高维直线
$$
\hat{y}^{(i)}=\theta_{0}+\theta_{1} X_{1}^{(i)}+\theta_{2} X_{2}^{(i)}+\ldots+\theta_{n} X_{n}^{(i)}
$$
为了向量化，给x添加一维特征，这一维特征的值恒为1
$$
\hat{y}^{(i)}=\theta_{0} X_{0}^{(i)}+\theta_{1} X_{1}^{(i)}+\theta_{2} X_{2}^{(i)}+\ldots+\theta_{n} X_{n}^{(i)}, X_{0}^{(i)} \equiv 1
$$
此时可以写成向量相乘的形式
$$
\hat{y}^{(i)}=X^{(i)} \cdot \theta
$$
数据集中共有m个样本  特征矩阵表示为
$$
X_{b}=\left(\begin{array}{ccccc}
1 & X_{1}^{(1)} & X_{2}^{(1)} & \ldots & X_{n}^{(1)} \\
1 & X_{1}^{(2)} & X_{2}^{(2)} & \ldots & X_{n}^{(2)} \\
\ldots & & & & \ldots \\
1 & X_{1}^{(m)} & X_{2}^{(m)} & \ldots & X_{n}^{(m)}
\end{array}\right)
$$
此时预测值的向量为
$$
\hat{y}=X_{b} \cdot \theta
$$
原损失函数
$$
\sum_{i=1}^{m}\left(y^{(i)}-\hat{y}^{(i)}\right)^{2}
$$
矩阵化
$$
\left(y-X_{b} \cdot \theta\right)^{T}\left(y-X_{b} \cdot \theta\right)
$$
最小化损失函数  得到多元线性回归的一般解的形式 称为多元线性回归的正规方程解
$$
\theta=\left(X_{b}^{T} X_{b}\right)^{-1} X_{b}^{T} y
$$
**采用正规方程优化损失函数时，不需要对数据做归一化处理，用原始数据即可，但是时间复杂度很高O(n^3)，而且矩阵不一定可逆**

### 3-4.python实现

#### 3-4-1.简单线性回归的python实现

```python
# 简单实现
import numpy as np

# 数据 numpy数组
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

# 按照最小二乘法的公式求a b
# 求出xy的均值
x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0.0  # a的分子
d = 0.0  # a的分母

# 遍历x和y中的每一个样本 按公式做累加
for x_i, y_i in zip(x, y):
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2
    
# 计算a b
a = num/d
b = y_mean - a * x_mean

# 向量化计算 快50倍
a = (x - x_mean).dot(y-y_mean)/(x-x_mean).dot(x-x_mean)  # 直接用*是对应元素相乘 不想加
b = y_mean - a * x_mean

# 拟合直线
y_hat = a * x + b

# 预测
x_predict = 6
y_predict = a * x_predict + b  # 5.2
```

按照sklearn格式封装成类

```python
import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None
	
    # 拟合数据集 其实就是求出参数 a b的过程
    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        self.a_=(x_train-x_mean).dot(y_train-y_mean)/(x_train-x_mean).dot(x_train-x_mean)
        self.b_ = y_mean - self.a_ * x_mean
        return self
    
    # 输入要预测样本的列表 返回预测值的列表
    def predict(self, x_predict):
        return np.array([self._predict(x) for x in x_predict])
	
    # 给定单个要预测的样本 返回预测值
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
	
    # 评价 用r square
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return 1 - (np.mean((y_predict-y_test)**2))/ np.var(y_test)

    # 打印实例对象时出现的类名
    def __repr__(self):
        return "SimpleLinearRegression()"
```

#### 3-4-2.多元线性回归的python实现

```python
# 封装成类
import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None  # 系数
        self.intercept_ = None  # 截距
        self._theta = None  # 总的参数

    def fit_normal(self, X_train, y_train):
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 特征矩阵
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)  # 正规方程

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self
    
    # 返回预测回归值的向量
    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)
	
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return 1 - (np.mean((y_predict-y_test)**2))/ np.var(y_test)

    def __repr__(self):
        return "LinearRegression()"

```

### 3-5.sklearn接口

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg.score(X_test, y_test)
```

### 3-6.线性回归的可解释性

```python
# 先创建一个线性回归实例拟合波士顿房价数据
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 查看一下系数  有正有负
lin_reg.coef_

# 系数按照从小到大排列并且取出对应的特征的名称
boston.feature_names[np.argsort(lin_reg.coef_)]
```

输出结果：['NOX', 'DIS', 'PTRATIO', 'LSTAT', 'CRIM', 'INDUS', 'AGE', 'TAX', 'B', 'ZN', 'RAD', 'CHAS', 'RM']

越右边的特征越与房价正相关  越左边越负相关

RM代表房间数 说明房间数越多 房价越贵

NOX代表一氧化碳浓度 说明一氧化碳浓度越高 房价越低

可解释性就是说人可以从拟合的模型中学到知识，是一个白盒模型

那么下一步就可以根据所学到的这些知识 进一步采集数据来优化模型 比如房间数对房价影响大 就可以采集更多房间数的特征 比如房间的类别 面积的大小 楼层的高低 等等

## 4.梯度下降

梯度下降不是一个机器学习算法 是一个基于搜索的最优化方法   用来最小化差距函数 

这里主要是以线性回归为例，用梯度下降法求解线性回归的损失函数的最小点

多维空间中导数是由多个分量的偏导组成的向量  称为梯度

### 4-1.批量梯度下降

**原理：**

每次参数减去损失对它的导数，更新得到的新参数会使损失函数比原来的变小

![image-20201210134910267](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201210134910267.png)

**特点：**

稳定 每次都朝下降最快的方向前进

耗时

**问题：**

1.并不是所有的函数都有唯一的极值点

解决方案：可以多次运行 随机初始化 

![image-20201210135101228](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201210135101228.png)

2.使用梯度下降时要做数据归一化   不做归一化 有的特征尺度大有的特征尺度小 得到的梯度乘一个正常的学习率 对于小尺度的特征 仍然会很大  难收敛  乘一个很小的学习率 对于大尺度的特征 需要很长的时间才能收敛 因此要做归一化

![image-20201210143514419](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201210143514419.png)

**求解：**

以线性回归为例，在线性回归中，我们选择的损失函数是平方误差SE:
$$
\sum_{i=1}^{m}\left(y^{(i)}-\hat{y}^{(i)}\right)^{2}
$$
其中预测值y_hat用参数表示为：
$$
\hat{y}^{(i)}=\theta_{0}+\theta_{1} X_{1}^{(i)}+\theta_{2} X_{2}^{(i)}+\ldots+\theta_{n} X_{n}^{(i)}
$$
那么损失函数就可以写成：
$$
\sum_{i=1}^{m}\left(y^{(i)}-\theta_{0}-\theta_{1} X_{1}^{(i)}-\theta_{2} X_{2}^{(i)}-\ldots-\theta_{n} X_{n}^{(i)}\right)^{2}
$$
用梯度下降法求它的最小值，就是每次theta都要沿着梯度的方向做更新

这里用损失函数对theta的每个分量求偏导
![](http://latex.codecogs.com/svg.latex?$$
\nabla J(\theta)=\left(\begin{array}{c}
\partial J / \partial \theta_{0} \\
\partial J / \partial \theta_{1} \\
\partial J / \partial \theta_{2} \\
\cdots \\
\partial \theta_{n}
\end{array}\right) \quad=\left(\begin{array}{c}
\sum_{i=1}^{m} 2\left(y^{(i)}-X_{b}^{(i)} \theta\right) \cdot(-1) \\
\sum_{i=1}^{m} 2\left(y^{i j}-X_{b}^{(i)} \theta\right) \cdot\left(-X_{1}^{(i)}\right) \\
\sum_{i=1}^{m} 2\left(y^{i \prime}-X_{b}^{(i)} \theta\right) \cdot\left(-X_{2}^{(i)}\right) \\
\cdots \\
\sum_{i=1}^{m} 2\left(y^{i n}-X_{b}^{(i)} \theta\right) \cdot\left(-X_{n}^{(i)}\right)
\end{array}\right)
=2 \cdot\left(\begin{array}{c}
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{1}^{(i)} \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{2}^{(i)} \\
\cdots \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{n}^{(i)}
\end{array}\right)
$$)
这里样本数m越大，每一个偏导也就越大，这是不合理的，因此每个偏导再除以样本数m
$$
\nabla J(\theta)=\left(\begin{array}{c}
\partial J / \partial \theta_{0} \\
\partial J / \partial \theta_{1} \\
\partial J / \partial \theta_{2} \\
\partial J / \partial \theta_{n}
\end{array}\right)
=\frac{2}{m}  \cdot\left(\begin{array}{c}
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{1}^{(i)} \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{2}^{(i)} \\
\cdots \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{n}^{(i)} \\
\end{array}\right)
$$
此时损失函数已经从平方误差SE变成了均方误差MSE：

有时也会取分母等于2m
$$
\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)}-\hat{y}^{(i)}\right)^{2}
$$
对这个公式做向量化：
$$
\nabla J(\theta)
=\left(\begin{array}{l}
\partial J / \partial \theta_{0} \\
\partial J / \partial \theta_{1} \\
\partial J / \partial \theta_{2} \\
\partial J / \partial \theta_{n}
\end{array}\right)
=\frac{2}{m} .\left(\begin{array}{c}
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{0}^{(i)} \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{1}^{(i)} \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{2}^{(i)} \\
\cdots \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{n}^{(i)}
\end{array}\right)
\\
=\frac{2}{m} \cdot\left(X_{b}^{(1)} \theta-y^{(1)}, \quad X_{b}^{(2)} \theta-y^{(2)}, \quad X_{b}^{(3)} \theta-y^{(3)}, \quad \ldots \quad X_{b}^{(m)} \theta-y^{(m)}\right.) \cdot
\left(\begin{array}{ccccc}
X_{0}^{(1)} & X_{1}^{(1)} & X_{2}^{(1)} & \ldots & X_{n}^{(1)} \\
X_{0}^{(2)} & X_{1}^{(2)} & X_{2}^{(2)} & \ldots & X_{n}^{(2)} \\
X_{0}^{(3)} & X_{1}^{(3)} & X_{2}^{(3)} & \ldots & X_{n}^{(3)} \\
\ldots & \ldots & \ldots & \ldots & \ldots \\
X_{0}^{(m)} & X_{1}^{(m)} & X_{2}^{(m)} & \ldots & X_{n}^{(m)}
\end{array}\right)
\\
=\frac{2}{m} \cdot\left(X_{b} \theta-y\right)^{T} \cdot X_{b}=\frac{2}{m} \cdot X_{b}^{T} \cdot\left(X_{b} \theta-y\right)
$$

### 4-2.随机梯度下降

**原理：**

当样本量特别大时 求梯度十分耗时  因此考虑每次求梯度不对所有样本求 而是对当前的这一个样本求梯度 

**特点：**

灵活   快速   **能跳出局部最优解**

但是不能保证每次都是下降的方向 

**问题：**

1.靠近最小点时  由于随机下降 可能会再次离开最小点

设置一个学习率，使它可以随迭代次数增多而变小 

![image-20201210144907738](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201210144907738.png)

**求解：**

对于批量下降，计算各个分量的偏导时，要求和所有样本，而随机梯度下降，只求随机样本x_i的偏导
$$
\nabla J(\theta)=\left(\begin{array}{c}
\partial J / \partial \theta_{0} \\
\partial J / \partial \theta_{1} \\
\partial J / \partial \theta_{2} \\
\partial J / \partial \theta_{n}
\end{array}\right)
=\frac{2}{m} \cdot\left(\begin{array}{c}
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{0}^{(i)} \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{1}^{(i)} \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{2}^{(i)} \\
\cdots \\
\sum_{i=1}^{m}\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{n}^{(i)}
\end{array}\right)
$$
此时可以去掉分母m   做向量化
$$
\nabla J(\theta)=2 \cdot\left(\begin{array}{l}
\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{0}^{(i)} \\
\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{1}^{(i)} \\
\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{2}^{(i)} \\
\cdots \\
\left(X_{b}^{(i)} \theta-y^{(i)}\right) \cdot X_{n}^{(i)}
\end{array}\right)=2 \cdot\left(X_{b}^{(i)}\right)^{T} \cdot\left(X_{b}^{(i)} \theta-y^{(i)}\right)
$$

### 4-3.小批量梯度下降

求导数时不取全部的样本 也不只取一个样本  而是取一个超参数batch作为求导时的样本数

### 4-4.python实现

#### 4-4-1.模拟梯度下降

```python
# 损失函数  学习率没设置好时 eta*gradient可能会很大 导致损失函数一直增加 超过float表达极限时会报错 如果报错 我们就取float所能表示的最大数inf
def J(theta):
    try:
   		return (theta-2.5)**2 - 1
    except:
        return float('inf')

# 损失函数的导数
def dJ(theta):
    return 2*(theta - 2.5)

theta = 0.0  # 初始化参数
eta = 0.01  # 学习率
epsilon = 1e-6  # 梯度最小值

# 梯度下降
while True:
    gradient = dJ(theta)  # 当前梯度
    last_theta=theta
    theta = theta - eta * gradient  # 更新参数
    if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
        break

print(theta)  # 2.499891109642585
print(J(theta))  # -0.99999998814289
```

可视化：

![image-20201210161339093](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201210161339093.png)

可以看到随着梯度变小，每一次theta的增量也变小

#### 4-4-2.批量梯度下降

这里求解的是线性回归问题

```python
# 求导函数 根据推导式可知 dJ = 2/m *  <(<X_b,theta>-y),X_b.T>
def dJ(theta, X_b, y):
    return 2. / len(X_b) * X_b.T.dot(X_b.dot(theta)-y)

# 梯度下降 
def gradient_descent(X_b, y, theta, eta=0.01, epsilon=1e-6, epochs=1e4):
	epoch = 0
    while epoch < epochs:
        gradient = dJ(theta, X_b, y)
        last_theta=theta
        theta = theta - eta * gradient
        if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
        epoch += 1
        
    return theta

X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 特征矩阵前面补一列1得到X_b
init_theta = np.zeros(X_b.shape[1])  # 初始化theta  size和特征数一致
theta = gradient_descent(X_b, y_train, init_theta)
```

#### 4-4-3.随机梯度下降

这里求解的是线性回归问题

```python
# 求导函数 随机梯度下降每次只用一个样本求损失函数 然后求theta各个分量的偏导 
def dJ(theta, X_b_i, y_i):
    return 2. * X_b_i.T.dot(X_b_i.dot(theta)-y_i)

# 随机梯度下降的学习率
def learning_rate(t):
    return t0 / (t + t1)

# 梯度下降 
def gradient_descent(X_b, y, theta, epochs=100, t0=5, t1=50):
	m = len(X_b)
    for epoch in range(epochs):
        # 打乱样本顺序
        indexes = np.random.permutation(m)
        X_b_new = X_b[indexes]
        y_new = y[indexes]
        # 每个epoch迭代所有样本一遍
        for i in range(m):
            gradient = dJ(theta, X_b_new[i], y_new[i])  # 当前样本的梯度
            theta = theta - learning_rate(epoch*m+i) * gradient
     return theta
     
X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 特征矩阵前面补一列1得到X_b
init_theta = np.zeros(X_b.shape[1])  # 初始化theta  size和特征数一致
theta = gradient_descent(X_b, y_train, init_theta)
```

#### 4-4-4.小批量梯度下降

```python
# 求导函数 随机梯度下降每次用batch个样本求导数
def dJ(theta, X_b_batch, y_batch):
    return 2./ len(X_b_batch) * X_b_batch.T.dot(X_b_batch.dot(theta)-y_batch)

# 随机梯度下降的学习率
def learning_rate(t):
    return t0 / (t + t1)

# 梯度下降 
def gradient_descent(X_b, y, theta, epochs=100, t0=5, t1=50, batch=4):
	m = len(X_b)
    step = m // batch  # 遍历所有样本要多少步
    for epoch in range(epochs):
        # 打乱样本顺序
        indexes = np.random.permutation(m)
        X_b_new = X_b[indexes]
        y_new = y[indexes]
        # 每个epoch迭代所有样本一遍
        for i in range(0, m, batch):
            if i+batch>m-1:
                last = m-1
            else:
                last = i+batch
            gradient = dJ(theta, X_b_new[i:last], y_new[i:last])  # 当前样本的梯度
            theta = theta - learning_rate(epoch*step + i // batch) * gradient
     return theta
     
X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 特征矩阵前面补一列1得到X_b
init_theta = np.zeros(X_b.shape[1])  # 初始化theta  size和特征数一致
theta = gradient_descent(X_b, y_train, init_theta)
```

### 4-5.sklearn接口

```python
# 解决线性回归的梯度下降法
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter=50)
sgd_reg.fit(X_train_standard, y_train)
sgd_reg.score(X_test_standard, y_test)
```

### 4-6.梯度验证

根据损失函数 数学推导求得梯度表达式  可以用梯度的定义来验证是否正确

![image-20201210171024129](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201210171024129.png)

取任一点附近两个点带入损失函数，然后除以两点距离，得到这一点得导数 (对于多维的情况  每次求一个方向的偏导)

用这个方法在部分样本上拟合

然后用梯度的数学推导式的梯度下降法在同样的样本上拟合  比较参数是否一致

不能直接用定义式求整体样本的参数  因为要求两次损失函数的值  太慢 数学推导式不需要求损失函数的值


$$
\frac{d J}{d \theta}=\frac{J(\theta+\varepsilon)-J(\theta-\varepsilon)}{2 \varepsilon}
$$

```python
import numpy as np

np.random.seed(666)
X = np.random.random(size=(1000, 10))
true_theta = np.arange(1, 12, dtype=float) 
X_b = np.hstack([np.ones((len(X), 1)), X])
y = X_b.dot(true_theta) + np.random.normal(size=1000)

# 线性回归的损失函数MSE
def J(theta, X_b, y):
    return np.mean((X_b.dot(theta)-y)**2)

# 导数数学推导式
def dJ_math(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

# 定义式
# 每次求一个方向的偏导 然后组合在一起  
def dJ_debug(theta, X_b, y, epsilon=0.01):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2[i] -= epsilon
        res[i] = (J(theta_1, X_b, y) - J(theta_2, X_b, y)) / (2 * epsilon)
    return res

# 梯度下降 
def gradient_descent(dJ, X_b, y, theta, eta=0.01, epsilon=1e-8, epochs=1e4):
    epoch = 0
    while epoch < epochs:
        gradient = dJ(theta, X_b, y)
        last_theta=theta
        theta = theta - eta * gradient
        if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
        epoch += 1
        
    return theta

# 初始化
X_b = np.hstack([np.ones((len(X), 1)), X])
theta = np.zeros(X_b.shape[1])

# 求解 然后比较参数
theta_math = gradient_descent(dJ_math, X_b, y, theta)
theta_debug = gradient_descent(dJ_debug, X_b, y, theta)

print(true_theta) 
print(theta_math)
print(theta_debug)
```

## 5.主成分分析PCA

主成分分析是一个非监督学习算法，不用标记，他的作用是把高维的数据降低到低维，比如说将二维数据降到一维，就是找一个轴，使得样本映射到这个轴上之后，间距最大，即方差最大，此时说明样本的差异性保持得最好，最能代表原来的数据分布

<img src="https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201225091851549.png" alt="image-20201225091851549" style="zoom:80%;" />

PCA的第一步是demean，即均值归0，此时方差更好计算
$$
Var(x)=\frac{1}{m} \sum_{i=1}^{m} (x_i-\bar{x})^2=\frac{1}{m} \sum_{i=1}^{m} x_i^2
$$
求解PCA问题，有三种解法：

1.梯度上升法  

2.基于特征值分解的协方差矩阵

3.基于SVD分解的协方差矩阵

后两种方法见  https://zhuanlan.zhihu.com/p/37777074/

PCA在降维时，去除了一些信息，这些信息有可能本身就是无用的，因此PCA有降噪的功能。

```python
# 降至二维  方便可视化
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

# 保存95% 以上信息
from sklearn.decomposition import PCA

pca = PCA(0.95)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
```

## 6.多项式回归和模型泛化

**原理：**

假设数据是非线性关系的  添加某些特征的多项式作为新的特征  添加后总的特征和回归值是线性相关的  可以用线性回归拟合新的数据  这种解决方式叫做多项式回归

比如有特征x 添加x的多项式x^2作为新的特征 此时可以拟合一条曲线 :
$$
y=a x^{2}+b x+c
$$
**特点：**

没有新的算法 就是为样本添加新的特征 然后用线性回归解决问题

和PCA相反  是对数据的一种升维  

### 6-1.python实现

要拟合的数据如图所示

![image-20201211134955043](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201211134955043.png)

```python
from sklearn.linear_model import LinearRegression
# 线性回归
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict = lin_reg.predict(X)

# 多项式回归：其实就是在线性回归前 做一次特征的扩充
X2 = np.hstack([X, X**2])  # 加入新的特征X**2得到新的特征矩阵x2  

lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)
```

效果图：

![image-20201211135305496](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201211135305496.png)

### 6-2.sklearn接口

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 添加新特征 ： 也属于数据的一种预处理 使用方法和归一化类似
poly = PolynomialFeatures(degree=3)  # 创建实例
poly.fit(X)  # 拟合数据
X2 = poly.transform(X)  # 扩充特征

# 线性回归
lin_reg = LinearRegression()
lin_reg.fit(X2, y)
y_predict2 = lin_reg.predict(X2)
```

这里degree=3意思是添加的特征多项式次数最大是3次方 

比如原来数据的特征有 x1 x2 当degree=3时 新的特征矩阵就会有10种特征

![image-20201211135536039](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201211135536039.png)

### 6-3.pipeline构造多项式回归函数

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# 构造pipeline之后 fit时会自动按照顺序对预处理的函数做fit_transform 然后对模型函数做fit
def poly_reg(degree):
	return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandarScaler()),
        ('reg', LinearRegression())
	])
reg = poly_reg(3)
reg.fit(X, y)
y_predict = reg.predict(X)  # 调用和普通模型一致
```

### 6-4.过拟合和欠拟合

左图是普通线性回归  没有很好的拟合数据 这是一种欠拟合 

欠拟合一般是因为选用的模型不能很好的表达数据间的关系(模型表达能力不足  对数据关系的假设步成立等)  导致预测的数据和真实数据**偏差**比较大

右图是degree=100时的多项式回归  它过度的拟合了数据  包括一些异常数据  这是一种过拟合

过拟合一般是因为模型太复杂  学习能力太强  数据的一点点扰动都会极大的影响模型  这使得模型对相似数据的预测可能会有很大的不同  **方差**比较大

![image-20201211140742248](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201211140742248.png)

非参数学习的机器学习算法都是高方差的  因为不对数据做任何假设  预测本身是基于训练数据的 训练数据本身的异常对预测结果干扰很大  因此方差高  这其实也是一种在训练集上的过拟合

参数学习的机器学习算法都是高偏差的  因为他对数据有极强的假设  当假设不成立时 就导致了很高的偏差 

**防止过拟合或者欠拟合 就是要调整方差和偏差**

线性回归模型复杂度低  偏差大 方差小  

degree=100的多项式回归模型复杂度高 偏差小 方差大 

调整方差和偏差相差不多时  就达到了比较好的拟合状态  

### 6-5.解决过拟合

机器学习中的欠拟合即高偏差 一般是来自于对数据的假设不成立 比如用以前的股市数据预测未来的股市数据  这可能本身就是行不通的 因此对数据的这种假设不在机器学习算法的讨论中

机器学习算法更关注于解决过拟合即高方差的问题 

解决高方差的方法：

1.降低模型复杂度  ： 降低学习能力

2.减少数据维度  降噪  ： 保留最有用得特征  减少异常点  减少扰动对模型的影响 

3.增加样本数  ： 更充分学习

4.使用交叉验证  ： 防止模型在单一的验证集上过拟合  

5.正则化  ： 通过损失函数抑制可学习参数的大小  来抑制模型的表达能力

### 6-6.交叉验证  

**随机分割数据可能导致训练集或者验证集中包含大量异常数据   交叉验证进行多次分割 可以减少这种随机性**

以多项式回归为例  degree=[2,3,4,5,6,7]  选择不同的degree有6种不同的模型

单一的验证集：从训练集中分割一部分做验证集  在训练集上得到6个不同的模型 我们用得到的6个模型在验证集上预测  取分数最高的这个模型  

交叉验证：将训练集分成k份 k-1份做训练集 在训练集上得到k-1个模型 然后再1份验证集上的到分数 平均后就是当前degree的模型的分数  然后选取得分最高的degree模型

**交叉验证实际上是减少了在分割训练集和验证集时的随机性   防止在过多的异常数据中拟合或者评分**  

**留一法**：

有m个样本 把训练集分m份 留一份做验证集 

这种方法完全不受随机的影响  最接近当前超参数的真实模型性能

但是每个超参数都要产生m-1个模型求平均 计算量太大 

```python
# 交叉验证
from sklearn.model_selection import cross_val_score

knn_clf = KNeighborsClassifier()
cross_val_score(knn_clf, X_train, y_train)  # 默认cv=3 返回[ 0.98895028,  0.97777778,  0.96629213] 代表当前模型在三个验证集上的得分

# 交叉验证选取超参
best_k, best_p, best_score = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        scores = cross_val_score(knn_clf, X_train, y_train)
        score = np.mean(scores)
        if score > best_score:
            best_k, best_p, best_score = k, p, score
            
print("Best K =", best_k)  # 2
print("Best P =", best_p)  # 2
print("Best Score =", best_score)  # 0.98

# 网格搜索用的就是交叉验证
from sklearn.model_selection import GridSearchCV
param_grid = [
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(2, 11)], 
        'p': [i for i in range(1, 6)]
    }
]

grid_search = GridSearchCV(knn_clf, param_grid, verbose=1)
grid_search.fit(X_train, y_train)

grid_search.best_score_  # 0.98
grid_search.best_params_  # 'weights' 'distance' 2  2 
grid_search.best_estimator_  # 返回最好的模型
```

### 6-7.正则化

**通过损失函数抑制可学习参数的大小  来抑制模型的表达能力**

#### 6-7-1.岭回归

更改损失函数：
$$
J(\theta)=M S E(y, \hat{y} ; \theta)+\alpha \frac{1}{2} \sum_{i=1}^{n} \theta_{i}^{2}
$$

```python
# sklearn接口
from sklearn.linear_model import Ridge

def RidgeRegression(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=alpha))
    ])
```

alpha无限大时  theta无限小 此时模型是一根平行于x轴的线

![image-20201211155042860](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201211155042860.png)

#### 6-7-2.LASSO回归

更改损失函数：
$$
J(\theta)=M S E(y, \hat{y} ; \theta)+\alpha \sum_{i=1}^{n}\left|\theta_{i}\right|
$$

```python
# sklearn接口
from sklearn.linear_model import Lasso

def LassoRegression(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lasso_reg", Lasso(alpha=alpha))
    ])
```

Lasso用的是绝对值表达theta的大小 没有用平方 alpha取0.01相当于取0.0001

![image-20201211155357618](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201211155357618.png)

#### 6-7-3.Ridge 和 Lasso 比较  

Ridge在alpha=100时仍是一条曲线 Lasso在alpha=0.1时已经基本变成了一条直线 

因为Ridge更倾向于保留所有theta  而Lasso更倾向于把一些theta快速的置为0

alpha相同时  若theta小于1  Ridge平方后变成一个更小的值  此时Ridge正则项的惩罚力度是小于Lasso的 theta每减小一点 alpha必须呈平方级的上升才能保证惩罚力度 因此Ridge更倾向于把theta都变成很小的值 而Lasso的theta变小时  alpha正常改变就能保持惩罚力度 因此Lasso更倾向于把theta置0

#### 6-7-4.L1 L2正则 

knn中提到过明可夫斯基距离
$$
\left(\sum_{i=1}^{n}\left|X_{i}^{(a)}-X_{i}^{(b)}\right|^{p}\right)^{\frac{1}{p}}
$$
类似的有一种Lp范式  表述的是原点到当前点的距离
$$
\|x\|_{p}=\left(\sum_{i=1}^{n}\left|x_{i}\right|^{p}\right)^{\frac{1}{p}}
$$
p=1时就是Lasso正则项 又叫L1正则项

p=2时就是Ridge正则项 又叫L2正则项

#### 6-7-5.弹性网

弹性网其实就是L1正则和L2正则的一个加权  引入了一个[0,1]之间的系数r
$$
J(\theta)=M S E(y, \hat{y} ; \theta)
+r \cdot \alpha \sum_{i=1}^{n}\left|\theta_{i}\right|
+(1-r) \cdot \frac{\alpha}{2} \sum_{i=1}^{n} \theta_{i}^{2}
$$

## 7.逻辑回归

### 7-1.介绍

解决分类问题

叫回归是因为 它预测的是属于某个类别的概率值

逻辑回归首先用回归算法得到一个值t，然后用sigmoid函数将这个值限制在[0,1]之间，即概率值
$$
\sigma(t)=\frac{1}{1+e^{-t}}
$$
回归值t>=0时  概率p>=0.5  认为是正类别  

回归值t<0时 概率p<0.5  认为是负类别  

这里0.5叫做分类阈值thresh

逻辑回归本身只能用来做二分类(knn天生可以做多分类)

![image-20201215143244689](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201215143244689.png)

逻辑回归采用交叉熵作为损失函数(knn没有损失函数)    
$$
\operatorname{cost}=-y \log (\hat{p})-(1-y) \log (1-\hat{p})
$$
y=1时 正样本 此时p越接近1损失越小

y=0时 负样本 此时p越接近0损失越小

![image-20201215143840334](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201215143840334.png)

假设有m个样本 此时m个样本的总损失值为:
$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log \left(\hat{p}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-\hat{p}^{(i)}\right)
$$
其中:
$$
\hat{p}^{(i)} = \sigma\left(X_{b}^{(i)} \theta\right) = \frac{1}{1+e^{-X_{b}^{(i)} \theta}}
$$
sigmoid中的值可以用线性回归得到也可以用多项式回归(其实也是线性回归)得到

因此这里按照线性回归考虑

但是这个损失函数没有解析解 只能用梯度下降求解

线性回归中损失的梯度向量化后得到:
$$
\nabla J(\theta)=\frac{2}{m} \cdot X_{b}^{T} \cdot\left(X_{b} \theta-y\right)
$$
推导后得到逻辑回归损失的梯度向量化后为：
$$
\nabla J(\theta)=\frac{1}{m} \cdot X_{b}^{T} \cdot\left(\sigma(X_{b} \theta)-y\right)
$$

### 7-2.python实现

```python
# sigmoid函数
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# 损失函数
def J(theta, X_b, y):
    p = sigmoid(X_b.dot(theta))
    return -np.mean((y*np.log(p) + (1-y)*np.log(1-p)))

# 损失函数导数
def dJ(theta, X_b, y):
    return X_b.T.dot(sigmoid*X_b.dot(theta)-y) / len(y)

# 批量梯度下降
def gradient_descent(X_b, y, theta, eta=0.01, epsilon=1e-6, epochs=1e4):
	epoch = 0
    while epoch < epochs:
        gradient = dJ(theta, X_b, y)
        last_theta=theta
        theta = theta - eta * gradient
        if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
        epoch += 1
        
    return theta

X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
theta = np.zeros(X_b.shape[1])
theta = gradient_descent(X_b, y_train, theta)
```

### 7-3.决策边界

回归值t>=0时  概率p>=0.5  认为是正类别  

回归值t<0时 概率p<0.5  认为是负类别

t=<X_b,  theta>=0  这条线叫做决策边界

因此<X_b, theta>这条线称为决策边界

```python
# 绘制决策边界  传入任意fit后的模型和特x0  x1  特征的最小最大值作为坐标系范围
def plot_decision_boundary(model, axis):    
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),)
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
    
plot_decision_boundary(log_reg, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![image-20201215151833025](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201215151833025.png)

### 7-4.sklearn接口

```python
"""
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
C：正则化  损失前面系数为C  正则项前面系数是1 C越大 正则项占比越少 系数越得不到限制 泛化能力变小
penalty：正则化类型
multi_class='ovr'：采用ovr的方式做多分类  方式是liblinear  如果要用ovo 可选multi_class="multinomial", solver="newton-cg"
"""
# 线性逻辑回归
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 多项式逻辑回归
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X_train, y_train)
```

### 7-5.二分类改多分类

#### 7-5-1.OVR ：one vs rest  

n个类别分类n次 每次判断是当前类别的概率  得到n个概率 选择最高的作为分类概率

![image-20201215153612286](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201215153612286.png)

#### 7-5-2.OVO：one vs one 

n个类别分类C(n,2)次 每次判断是两个类别中的哪一个  最终选择得票数最高的分类

![image-20201215153926252](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201215153926252.png)

```python
# 逻辑回归中封装的OVR OVO(具体见上一节sklearn接口)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()  # ovr
log_reg2 = LogisticRegression(multi_class="multinomial", solver="newton-cg")  # ovo

# sklearn中的ovr ovo类
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

ovr = OneVsRestClassifier(cls_model) # 把一个二分类器封装成OVR多分类器
ovr.fit(X_train, y_train)
ovr.score(X_test, y_test)

ovo = OneVsOneClassifier(cls_model)  # 把一个二分类器封装成OVO多分类器
ovo.fit(X_train, y_train)
ovo.score(X_test, y_test)
```

## 8.分类任务评价标准

回归任务评价标准：MSE  MAE  RMSE  R2

分类任务评价标准：准确度accuracy

准确度在类别不平衡时不适用  比如分类男女 但是样本中99.99%都是女性  那么分类器只要所有样本都预测为女性 也能达到99.99%的准确率  但这样的分类器是不可泛化的

### 8-1.混淆矩阵，精确率，召回率

对于一个n分类任务 混淆矩阵是一个nxn的二维矩阵

以二分类0/1为例  混淆矩阵是一个[2，2]的二维矩阵 第一个维度是真实值  第二个维度是预测值

N/P代表预测是0/1   

T/F代表预测是正确/错误

|              | 预测值=0 | 预测值=1 |
| :----------: | :------: | :------: |
| **真实值=0** |    TN    |    FP    |
| **真实值=1** |    FN    |    TP    |

精确率 precision = TP / (FP+TP)   预测是该类别的样本里面到底有多少真的是该类别

召回率recall = TP / (FN+TP)   所有该类别的样本有多少被预测出来了

#### 8-1-1.python实现

```python
# 预测成负样本 预测正确
def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

# 预测成正样本 预测错误
def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

# 预测成负样本 预测错误
def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

# 预测成正样本 预测正确
def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

# 二分类的混淆矩阵
def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])
```

#### 8-1-2.sklearn接口

```python
from sklearn.metrics import confusion_matrix,precision_score,recall_score

confusion_matrix(y_test, y_log_predict)  # 混淆矩阵 适用于多分类
precision_score(y_test, y_log_predict)  # 适用二分类 多分类任务修改参数average
recall_score(y_test, y_log_predict)  # 适用二分类 多分类任务修改参数average
```

#### 8-1-3.多分类的混淆矩阵

用逻辑分类在手写数字集上求混淆矩阵并画出

```python
cfm = confusion_matrix(y_test, y_predict)
plt.matshow(cfm, cmap=plt.cm.gray)  # 可以直接imshow显示热力图
plt.show()
```

![image-20201215172748027](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201215172748027.png)

图中越亮的地方代表值越大，这里对角最亮，说明预测正确的比较多

将对角线置0，然后其他项都归一化，查看分类错误的情况

```python
# 归一化
row_sums = np.sum(cfm, axis=1)
err_matrix = cfm / row_sums 

# 置0
np.fill_diagonal(err_matrix, 0)

plt.matshow(err_matrix, cmap=plt.cm.gray)
plt.show()
```

![image-20201215172958269](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201215172958269.png)

可以看到1预测成9和8预测成1的错误情况最多

### 8-2.F1 Score 

具体更关注精准率还是召回率 应该根据场景来  比如绝缘子缺陷就应该更关注召回率

降低阈值可以提高召回率  提高阈值可以增加精准率

如果没有要求 那么我们希望两者能够有一个平衡 

F1 Score  两者的调和平均值  可以用来衡量两者是否平衡   也可以用来衡量分类的效果
$$
\frac{1}{F 1}=\frac{1}{2}\left(\frac{1}{\text {precision}}+\frac{1}{\text {recall}}\right)
$$
化简
$$
F 1=\frac{2 \cdot \text { precision } \cdot \text { recall }}{\text { precision }+\text { recall }}
$$

```python
def f1_score(precision, recall):
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0  # 分母为0时
    
# sklearn接口
from sklearn.metrics import f1_score

f1_score(y_test, y_predict)
```

通过调整决策边界(阈值自动跟着调整)，调整precision和recall，从而调整F1

```python
# 加载切分数据集得到 X_train, X_test, y_train, y_test
# 创建拟合逻辑回归模型得到 log_reg

# decision_function得到进入sigmoid函数之前的回归值 回归值小于0时 sigmoid之后小于阈值0.5 负样本
decision_scores = log_reg.decision_function(X_test)  

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

precisions = []
recalls = []
f1 = []
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
for threshold in thresholds:
    y_predict = np.array(decision_scores >= threshold, dtype='int')
    precisions.append(precision_score(y_test, y_predict))
    recalls.append(recall_score(y_test, y_predict))
    f1.append(f1_score(y_test, y_predict))
	
plt.plot(thresholds, precisions, label='precision')
plt.plot(thresholds, recalls, label='recall')
plt.plot(thresholds, f1, label='f1')
plt.legend()
plt.show()
```

![image-20201216093020681](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201216093020681.png)

可以看出在决策边界在0左右时精准率和召回率相等，得到平衡，此时的f1值也最高

如果要更考虑召回率可以小于0，更考虑精准度可以大于0

因此F1值可以很好的帮我们选择决策边界和阈值

### 8-3.PR曲线

按照8-2中获得不同阈值的precision和recall，然后绘制pr曲线

![image-20201216091926676](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201216091926676.png)

从PR曲线也可以看出精准度大概0.9时出现拐点，此时达到了平衡，因为再增加精准度的话，召回率断崖式下降

如果一条PR曲线被另一条PR曲线包住，此时外面曲线的每一点都是优于里面曲线的，因此可以说外面曲线对应的二分类器效果更好

如果存在交叉，可以适用面积来判断，但更常用的是平衡点或者是F1值，平衡点越靠近右上角二分类器越好，F1值越大越好

![image-20201216094121349](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201216094121349.png)

```python
# sklearn接口
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)

# 画precision和recall变化曲线
plt.plot(thresholds, precisions[:-1])  # precisions最后一个数是1
plt.plot(thresholds, recalls[:-1])  # recall最后一个数是0
plt.show()

# 画pr曲线
plt.plot(precisions, recalls)
plt.show()
```

### 8-4.ROC曲线

一般判断二分类器的好坏，ROC曲线使用比PR曲线更多

横轴  有多少真值为0的被预测成1  FP/(FP+TN)     FPR

纵轴  有多少真值为1的被预测成1  TP/(TP+FN)     TPR

根据不同阈值下的FPR和TPR绘制ROC曲线

```python
def FPN(y_true, y_predict):
    FP = np.sum((y_predict==1) & (y_true==0))
    TN = np.sum((y_predict==0) & (y_true==0))
    return FP / (FP + TN)

def TPN(y_true, y_predict):
    TP = np.sum((y_predict==1) & (y_true==1))
    FN = np.sum((y_predict==0) & (y_true==1))
    return TP / (TP + FN)

fprs = []
tprs = []
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
for threshold in thresholds:
    y_predict = np.array(decision_scores >= threshold, dtype='int')
    fprs.append(FPR(y_test, y_predict))
    tprs.append(TPR(y_test, y_predict))
    
plt.plot(fprs, tprs)
plt.show()

# sklearn接口
from sklearn.metrics import roc_curve

fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
plt.plot(fprs, tprs)
plt.show()
```

![image-20201216101224309](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201216101224309.png)

左上角点(0,1)代表FPR=0,TPR=1,此时分类完全正确

### 8-5.AUC

auc是roc曲线下的面积，可以用来评价模型好坏

area under curve

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, decision_scores) 
print(auc)
```

## 9.支撑向量机SVM

### 9-1.简介

分类问题中，决策边界往往不止一个，支撑向量就是指离决策边界最近的向量，要求决策边界离支撑向量距离最远的算法叫做支撑向量机

![image-20201223091431750](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201223091431750.png)

### 9-2.最优化

d是决策边界到支撑向量的距离

margin=2d  

![image-20201223091724956](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201223091724956.png)

解析几何中点(x,y)到直线Ax+By+C=0的距离
$$
\frac{|A x+B y+C|}{\sqrt{A^{2}+B^{2}}}
$$
n维空间中，向量x到n维直线w.T x+b=0的距离
$$
\frac{\left|w^{T} x+b\right|}{\|w\|}
$$
对于平面中的任意一个向量都应该满足
$$
\left\{\begin{array}{cc}
\frac{w^{T} x^{(i)}+b}{\|w\|} \geq d & \forall y^{(i)}=1 \\
\frac{w^{T} x^{(i)}+b}{\|w\|} \leq-d & \forall y^{(i)}=-1
\end{array}\right.
$$
使w模长为1/d，那么化简可得，此时可看作支撑向量距离决策边界距离为1，因此这个条件中包含了对w模长的限制
$$
y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1
$$
对于任意支撑向量|w.T x + b| =1，因此
$$
\max \frac{\left|w^{T} x+b\right|}{\|w\|}
=\max \frac {1}{\|w\|}
=\min \|w\|
=\min \frac{1}{2}\|w\|^2
\\
st.y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1
$$

### 9-3.软间隔

硬间隔是指所有样本必须在支撑向量边界的一侧

软间隔即允许一些样本点跨越支撑向量边界甚至是决策边界

![image-20201223094831535](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201223094831535.png)

于是每个向量都引入一个松弛变量，优化问题变为
$$
\min \frac{1}{2}\|w\|^2
\\
st.y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1- \xi_i
$$
将松弛变量以L1正则化的形式引入
$$
\begin{aligned}
\min\limits_{\boldsymbol{w}, b,\boldsymbol{\xi}} & \;\; \frac12 ||\boldsymbol{w}||^2  + C \,\sum\limits_{i=1}^m \xi_i \\[1ex]
{\text { s.t. }} & \;\; y_{i}\left(\boldsymbol{w}^{\top} \boldsymbol{x}_{i}+b\right) \geq 1 - \xi_i \ \ \ \ \ \ \ \ \ \xi_i  \geq 0
\end{aligned}
$$

### 9-4.sklearn接口

涉及距离计算，要标准化

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC  

standardScaler = StandardScaler()
standardScaler.fit(X)
X_standard = standardScaler.transform(X)

svc = LinearSVC(C=1e9)  # 正则化因子C越大 越限制松弛变量  间隔越硬  泛化能力越弱 
svc.fit(X_standard, y)
```

### 9-5.多项式特征的svm

对于决策边界是曲线，可以引入多项式特征

可以直接自己设置多项式特征，也可以使用多项式特征的核函数

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline

# 设置多项式特征
def PolynomialSVC(degree, C=1.0):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("linearSVC", LinearSVC(C=C))
    ])

poly_svc = PolynomialSVC(degree=3)
poly_svc.fit(X, y)

# 使用多项式特征的核函数
def PolynomialKernelSVC(degree, C=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("kernelSVC", SVC(kernel="poly", degree=degree, C=C))
    ])

poly_kernel_svc = PolynomialKernelSVC(degree=3)
poly_kernel_svc.fit(X, y)
```

### 9-6.核函数

对于向量点乘，都可以使用核技巧，核技巧中用到的不同函数统称为核函数

SVM原优化函数为
$$
\begin{array}{l}
\min \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{m} \zeta_{i} \\
\text { st. } y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1-\zeta_{i} \\
\quad \zeta_{i} \geq 0
\end{array}
$$
用拉格朗日乘子求解得到
$$
\begin{array}{l}
\max \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i} x_{j} \\
\text { st. } 0 \leq \alpha_{i} \leq C \\
\quad \sum_{i=1}^{m} \alpha_{i} y_{i}=0
\end{array}
$$
其中x_i点乘x_j可以用到核技巧

在多项式特征中，我们得到x_i的多项式特征x_i_hat，x_j的多项式特征x_j_hat

在使用新特征(包含旧特征)的过程中，原来x_i点乘x_j，变成了x_i_hat点乘x_j_hat

为了跳过产生多项式特征和用多项式特征分别点乘的繁琐步骤，我们希望得到一个函数可以直接得到最后x_i_hat点乘x_j_hat的结果，这个函数就是多项式的核函数
$$
K\left(x^{(i)}, x^{(j)}\right)=x^{\prime(i)} x^{\prime(j)}
$$
核技巧就是如果要对一个或多个点乘向量做变换，可以用一个核函数直接得到变换后的点乘结果，根据变换的不同，核函数也不同

引入核技巧后，优化问题变为
$$
\begin{array}{l}
\max \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j}K\left(x^{(i)}, x^{(j)}\right) \\
\text { st. } 0 \leq \alpha_{i} \leq C \\
\quad \sum_{i=1}^{m} \alpha_{i} y_{i}=0
\end{array}
$$
对于二次方的多项式核函数来说
$$
\begin{align}
\begin{array}{l}
K(x, y)&=(x \cdot y+1)^{2} \\
K(x, y) &=\left(\sum_{i=1}^{n} x_{i} y_{i}+1\right)^{2} 
\\
&=\sum_{i=1}^{n}\left(x_{i}^{2}\right)\left(y_{i}^{2}\right)+\sum_{i=2}^{n} \sum_{j=1}^{i-1}\left(\sqrt{2} x_{i} x_{j}\right)\left(\sqrt{2} y_{i} y_{j}\right)+\sum_{i=1}^{n}\left(\sqrt{2} x_{i}\right)\left(\sqrt{2} y_{i}\right)+1 \\
&=x^{\prime} \cdot y^{\prime}\ \ 其中\ \ \ x^{\prime}=\left(x_{n}^{2}, \ldots, x_{1}^{2}, \sqrt{2} x_{n} x_{n-1}, \ldots, \sqrt{2} x_{n}, \ldots \sqrt{2} x_{1}, 1\right)
\end{array}
\end{align}
$$
一般性的多项式核函数为
$$
\begin{array}{l}
K(x, y)&=(x \cdot y+c)^{d} \\
\end{array}
$$
线性核函数就是
$$
\begin{array}{l}
K(x, y)&=x \cdot y
\end{array}
$$
SVM中用得最多的是高斯核函数，又叫径向基函数RBF

可以将每一个样本点(向量)映射到无穷维的空间

一般是m\*n维数据升维到m\*m维   开销大  但是对于m<n的情况很好用
$$
K(x, y)=e^{-\gamma\|x-y\|^{2}}
$$

```python
# sklearn接口
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def RBFKernelSVC(gamma):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma=gamma))
    ])

svc = RBFKernelSVC(gamma)  # gamma越大 模型越复杂 越倾向于过拟合
svc.fit(X, y)
```

![image-20201223105522189](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201223105522189.png)

### 9-7.解决回归问题

提前指定间隔距离，虚线直线的向量越多，表示中间的实线越能表达回归直线

![image-20201223112906245](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201223112906245.png)

```python
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('linearSVR', LinearSVR(epsilon=epsilon))
    ])

svr = StandardLinearSVR()
svr.fit(X_train, y_train)

svr = SVR(kernel="rbf", gamma=gamma, epsilon=epsilon)
svr.fit(X_train, y_train)
```

## 10.决策树

这里介绍的决策树都是CART  (cls and reg tree)  在d维度按值v划分

另外还有ID3 C4.5 C5.0等决策树

非参数学习    天然可以解决多分类问题   也可以解决回归问题   有很好的可解释性

非参数学习天然的过拟合   剪枝

一棵决策树：

![image-20201223153508642](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201223153508642.png)

对应的决策边界：

![image-20201223153812257](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201223153812257.png)

问题：每个结点在哪个特征上划分？每个特征在哪个值划分？  

以信息熵降低为标准

### 10-1.信息熵

信息熵代表数据的离散程度，信息熵越大表示离散程度越高
$$
H=-\sum_{i=1}^{k}p_{i}\ log(p_{i})
$$
(1,0,0)此时数据都属于第一类，不离散，信息熵为0

(0.1,0.2,0.7)此时数据三个类都有，较离散，信息熵为0.8018

对于二分类信息熵可以写成
$$
H=-xlog(x)-(1-x)log(1-x)
$$
如果当前结点在d维度按照值v划分使得**信息熵降低最多**，就按它划分

### 10-2.基尼系数

评价数据离散程度的另一个指标  [0,1]之间 基尼系数越大 离散程度越高
$$
G=1-\sum_{i=1}^{k}p^2
$$
二分类问题中
$$
G=-2x^2+2x
$$
基尼系数的计算要比信息熵快，因此sklearn中默认使用基尼系数，但是两者效果没有优劣之分

### 10-4.sklearn接口

```python
"""
DecisionTreeClassifier(
class_weight=None,   类别权重
criterion='gini',    分裂标准
max_depth=None,      最大树深
max_features=None,   一个结点最多特征的个数
max_leaf_nodes=None, 最大叶子节点数
min_impurity_decrease=0.0, 
min_impurity_split=None,
min_samples_leaf=1,  产生叶子结点要求最小的样本数
min_samples_split=2, 分裂要求的最小的样本数
min_weight_fraction_leaf=0.0, 
presort=False, 
random_state=None,
splitter='best')
"""
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(**kwarg)
dt_clf.fit(X, y)
```

### 10-5.解决回归问题 

先用特征做分类  得到一棵决策树

分类后叶子结点平均值作为回归值

同样是非参数学习算法

```python
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
```

### 10-6.局限性

决策树按照值划分，因此决策边界必须是平行于轴的

决策树是一个非参数学习算法  训练的过程其实是对根据训练集得到一个划分方法  因此天然的是过拟合的 这也是非参数学习算法的通病

决策树作为非参数学习算法  对异常点非常敏感 可能因为个别异常点导致算法的效果急剧下降

单个决策树的准确率并不高

## 11.集成学习和随机森林

### 11-1.简介

ensemble learning可以集成KNN 逻辑回归  SVM  决策树 神经网络 贝叶斯等诸多模型 

在训练集上训练多个不同的模型，然后进行投票，按照少数服从多数的方式确定最终预测的结果

这种少数服从多数的投票方式又叫硬投票

```python
# 集成逻辑回归 决策树和支撑向量机
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('log_clf', LogisticRegression()), 
    	('svm_clf', SVC()),
    	('dt_clf', DecisionTreeClassifier(random_state=666))
    ],
    voting='hard'
)

voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
```

### 11-2.软投票

|        |  A   |  B   |
| :----: | :--: | :--: |
| 模型一 | 0.99 | 0.01 |
| 模型二 | 0.49 | 0.51 |
| 模型三 | 0.4  | 0.6  |
| 模型四 | 0.9  | 0.1  |
| 模型五 | 0.3  | 0.7  |

按照硬投票 A:B=2:3  应该预测为B

按照软投票 A=(0.99+0.49+0.4+0.9+0.3)/5=0.616      B=(0.001+0.51+0.6+0.1+0.7)/5=0.384   应该预测为A

代码改变VotingClassifier的voting参数即可

### 11-3.bagging和pasting

集成学习要求子模型有差异    否则集成没有意义

模型种类有限 不能构建很多不同的子模型 为了得到不同的子模型 用部分数据去创建同一种模型

模型一般选择决策树  因为它有很多参数 可以保证子模型的差异

而选取数据的方式有两种：

常用的是放回取样 bagging   统计学中叫bootstrap

还有一种是不放回取样 pasting   这种方式训练的模型也有限  用的少

### 11-4.sklearn接口

```python
"""
base_estimator,		子模型
bootstrap=True, 	是否放回取样 
bootstrap_features=False, 	是否对特征进行放回采样
max_features=1.0,	一个子模型采样的最大特征数
max_samples=100,  	一个子模型采样的最大样本数
n_estimators=500, 	子模型个数
n_jobs=1, 			并行程度
oob_score=True,		out-of-bag 根据统计学 使用放回取样时 总样本中有37%的数据是没有取到的 因此可以在取样过程中记录取到了的数据 最后直接用没取过的做测试集 得到oob分数
random_state=None, 	
verbose=0, 
warm_start=False
"""
bagging & pasting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# 用决策树做子模型 取样500次 每次取100个样本 是否放回bootstrap=True就是bagging False就是pasting
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                           n_estimators=500, max_samples=100,
                           bootstrap=True)
bagging_clf.fit(X_train, y_train)
bagging_clf.score(X_test, y_test)
```

### 11-5.随机森林和Extra-Trees

用决策树做子模型，降低离散程度为标准分裂，得到的集成模型叫做随机森林

用决策树做子模型，随机分裂，得到的集成模型叫做Extra-Trees，它的随机程度更高，但是bias更大

```python
# 随机森林
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=666, n_jobs=-1)
rf_clf.fit(X, y)

# Extra-Trees
from sklearn.ensemble import ExtraTreesClassifier

et_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=666, n_jobs=-1)
et_clf.fit(X, y)
```

etra-tree

### 11-7.提升方法

除了随机创建子模型以外  有一种创建子模型的策略叫做提升方法boosting

bossting希望新产生的每个模型都能增强模型

#### 11-7-1.ada boosting

先拟合原始数据集，得到一个子模型，对于预测成功的点(浅色点)减少权重，对于预测错误的点(深色点)增加权重，然后拟合新数据又得到一个子模型，以此类推，可以得到n个子模型，然后集成这n个子模型进行投票预测

![image-20201223172622797](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201223172622797.png)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2), n_estimators=500)
ada_clf.fit(X_train, y_train)
```

#### 11-7-2.gradient boosting

第一个绿线图是第一个子模型m1，此时有一些样本点预测错误，这些样本点的真值减去预测值得到新的数据

第二个绿线图是拟合了新数据的子模型m2，此时总的模型是m1和m2的预测值相加，即第二个红图

类推可以得到最终的模型m=m1+m2+...+mn

![image-20201223173409644](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201223173409644.png)

```python
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=30)
gb_clf.fit(X_train, y_train)
```

#### 11-7-3.stacking

训练数据分两份

第一份训练三个子模型 三个子模型的输出再作为另一个模型的输入 

另一份数据训练最后集成子模型的融合模型

![image-20201224094211603](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201224094211603.png)

可以更复杂的把数据分成三份

![image-20201224095039912](https://github.com/zk2ly/Leaning_notes/blob/main/python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/README_IMG/image-20201224095039912.png)

因此stacking的层数和每层的模型数都是超参数

这里和神经网络很像，但是神经网络每个神经元是一个线性变化，stacking每一个结点是一个子模型

sklearn中没有stacking的实现

### 11-8.总结

对于集成模型 重点是要集成很多不同的子模型

对于子模型的产生方法，有一种是对于数据进行划分，有bagging和pasting两种方式

有一种是不断完善原来的子模型，叫做boosting方法

如果用决策树做基准模型  对数据进行划分来产生子模型  这种集成模型叫做随机森林 

如果每棵树的创建都不用离散度做分裂标准 那么叫做Extra-Trees
