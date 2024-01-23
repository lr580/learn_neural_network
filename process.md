## 1. data preprocessing

阅读代码可知，下面函数被 `util.py` 的 `load_data` 调用且尚未实现：

- `one_hot_encoding`
- `normalize_data`
- `createTrainValSplit`

### one hot

不使用 for 循环的优化，

```python
def one_hot_encoding(labels, num_classes=10):
    oneHot = np.zeros((labels.size, num_classes))
    oneHot[np.arange(labels.size), labels] = 1
    return oneHot
```

实践表明，还是太慢了(第3行导致的)，所以应该提高速度：

- (n,1) -> (n) 使用 flatten 然后 -> (n,10)

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
    return np.eye(num_classes)[labels.flatten()]
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



## 2. softmax regression

### Activation

传入的 x 是二维 numpy，维度分别为：批的大小(batch size)和特征数目。是加权和，即上一层的 $\sum wx$。

sigmoid：$\sigma(x)=\dfrac1{1+e^{-x}},\sigma'(x)=\sigma(x)(1-\sigma(x))$

tanh: $\tanh'(x)=1-\tanh^2(x)$

ReLU: $\max(0,x)$，导数 $x>0\to1$ 否则 $0$。

output / softmax: $\dfrac{e^{x_i}}{\sum_j e^{x_j}}$

具体计算：(避免溢出) 分子分母除以常数 $e^{\max(x)}$，结果为：$\dfrac{e^{x_i-\max(x)}}{\sum_j e^{x_j-max(x)}}$

因为 $e^x > 0$，所以分母不可能为零。

导数特殊计算，故暂时忽略，设置为 $1$。

根据公式写出代码即可。

```python
class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def ReLU(self, x):
        return np.maximum(0, x)

    def output(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def grad_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def grad_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def grad_ReLU(self, x):
        return np.where(x > 0, 1, 0)

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """
        return 1  #Deliberately returning 1 for output layer case
```



### Layer

### 报告

训练过程：如上代码所示。

超参数：默认。

