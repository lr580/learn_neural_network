## 1. data preprocessing

阅读代码可知，下面函数被 `util.py` 的 `load_data` 调用且尚未实现：

- `one_hot_encoding`
- `normalize_data`
- `createTrainValSplit`

### one hot

不使用 for 循环的优化，

```python
def one_hot_encoding(labels, num_classes=10):
    """
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (10 for MNIST)

    returns:
        oneHot : N X num_classes 2D array
    """
    oneHot = np.zeros((labels.size, num_classes))
    oneHot[np.arange(labels.size), labels] = 1
    return oneHot
```

### normalize

代码：

```python
def normalize_data(inp):
    """
    Normalizes image pixels here to have 0 mean and unit variance.

    args:
        inp : N X d 2D array where N is the number of examples and d is the number of dimensions

    returns:
        normalized inp: N X d 2D array

    """
    mean = np.mean(inp, axis=0)
    std = np.std(inp, axis=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        normalized_inp = np.where(std == 0, 0, (inp - mean) / std)
    return normalized_inp
```

这里特判了分母零，但是不一定需要特判，可以删掉特判，直接返回 `inp - mean) / std`，则分母 0 时返回 nan。

对拍验证正确性：

```python
from scipy import stats
stats.zscore(a,axis=0)
```

> 注意到 std 是除以 n，即总体标准差，如果要除以 n-1 (样本标准差)，要 `np.std(data, ddof=1) `

### 数据集划分

直接使用 numpy 随机打乱即可：

```python
def createTrainValSplit(x_train,y_train):

    """
    Creates the train-validation split (80-20 split for train-val). Please shuffle the data before creating the train-val split.
    """
    assert len(x_train) == len(y_train)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    split_index = int(0.8 * len(x_train))

    x_train_split = x_train_shuffled[:split_index]
    y_train_split = y_train_shuffled[:split_index]
    x_val_split = x_train_shuffled[split_index:]
    y_val_split = y_train_shuffled[split_index:]

    return x_train_split, y_train_split, x_val_split, y_val_split
```



### 报告

> 为了可复现，main 里固定随机种子：
>
> ```python
> np.random.seed(42)
> ```



数据集：名称、大小、简介。[here](https://paperswithcode.com/dataset/mnist) 从这里介绍

描述上面编写的代码(标准化实现、划分实现)的文字描述

求任意训练集的均值方差。

- 测试：在 `main.py` 的 `util.load__data` 后面加：

  ```python
  image = x_train[0]
  print(np.mean(image), np.std(image))
  # print(image[:30])
  # print(x_train.shape, y_train.shape)
  # print(x_valid.shape, y_valid.shape)
  # print(x_test.shape, y_test.shape)
  return 0
  ```

输出：

```
-0.044059613984173816 0.8928353591476925
```



## 2. 
