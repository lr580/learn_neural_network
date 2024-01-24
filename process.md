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

`neuralnet.py` 包含：

- `Activation` 类
- `Layer` 类
- `Neuralnetwork` 类

`train.py` 包含：

- `train` 方法
- `modelTest` 方法

### Activation

传入的 x 是二维 numpy，维度分别为：批的大小(batch size)和特征数目。是加权和，即上一层的 $\sum wx$。

sigmoid：$\sigma(x)=\dfrac1{1+e^{-x}},\sigma'(x)=\sigma(x)(1-\sigma(x))$

tanh: $\tanh'(x)=1-\tanh^2(x)$

ReLU: $\max(0,x)$，导数 $x>0\to1$ 否则 $0$。

output / softmax: $\dfrac{e^{x_i}}{\sum_j e^{x_j}}$

具体计算：(避免溢出) 分子分母除以常数 $e^{\max(x)}$，结果为：$\dfrac{e^{x_i-\max(x)}}{\sum_j e^{x_j-max(x)}}$

因为 $e^x > 0$，所以分母不可能为零。

导数特殊计算，故暂时忽略，设置为 $1$。

根据公式写出代码即可。(没改动的不展示)

```python
class Activation():
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def ReLU(self, x):
        return np.maximum(0, x)

    def output(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def grad_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def grad_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def grad_ReLU(self, x):
        return np.where(x > 0, 1, 0)
```

- 其中 axis=1 是逐行求和，对 (n, 10) 返回维度 (n)
- 增加 keepdims 返回 (n, 1)



### Layer

> 初始化方法增加一行动量相关的代码，下文后续再解释，本题用不到。
>
> ```python
> self.prev_dw = np.zeros_like(self.w)
> ```

前向传播：

- 定义权重矩阵里每列最后一个元素是偏置项 `b` (也可以定义头一个，但其它代码也要改)

- 因此对原始输入矩阵 x (维度为 `(N, 784)`)，修改为 `(N, 785)`，其中 28x28 MNIST 灰度图像 flatten 过(预处理做了)为 784

- 将增广矩阵 x 与本层权重矩阵 w (维度为 `(in_units+1,out_units)`)点乘，即 w 的维度是 `(785, 10)`，`(i,j)` 表示输入的第 i 个元素对输出的的 j 个元素的加权贡献是 `w[i,j]`

  故将 x,w 做矩阵乘法即可，将得到 a `(N, 10)`

- 然后对 a 做激活，得到维度不变的 z `(N,10)`

```python
def forward(self, x):
    self.x = np.hstack([x, np.ones((x.shape[0], 1))])
    self.a = np.dot(self.x, self.w)
    self.z = self.activation(self.a)
    return self.z
```

反向传播：（暂不考虑正则化、动量）

> - 设损失函数对下一层的 `a` 求偏导的结果是 `deltaCur`，即 $\dfrac{\partial E}{\partial a_{next}}$，大小为 (样本数, 下一层神经元数)
>
>   这个结果最开始为输出层的，并不断将结果返回给上一层；如最开始为 `(N, 10)`，
>
>   使用交叉熵损失函数为 $E(t,y)$，其中 $t$ 是预测输出，$y$ 是真实输出，其中 $t=softmax(a)$，其中 $a_{next}$ 就是 softmax，$a$ 是上一层 forward 的返回值，根据 PA1 Page3 2a 可知，最初始 $\dfrac{\partial E}{\partial a_{next}}=t-y$
>
> - 设权重更新量为 `w -= learning_rate * dw`，这个 dw 是损失函数对当前矩阵 w 的偏导，即 $\dfrac{\partial E}{\partial w}$
>
>   如果没有隐藏层(这一问)，只有输入到输出，那就是 $x\to a\to t$，即激活为 output (softmax)，即 $E(t,y),t=softmax(a),a=XW$，单独考虑函数 $a$，可以求导 $\dfrac{\partial a}{\partial w}=x^T$，即 `(785, N)` 形状
>
>   因为已经求得 $\dfrac{\partial E}{\partial a_{next}}$，故 $\dfrac{\partial E}{\partial w}=\dfrac{\partial E}{\partial a_{next}}\dfrac{\partial a_{next}}{\partial w}$，即 `(785, N)` 矩阵乘以 `(N, 10)` 矩阵，得到与 w 一样的维度的偏导结果
>
>   此时可以更新权重
>
> - 对输出层，定义 output 的偏导是 1 (忽略不计)，所以，如果这一层(倒数第一个 Layer)往前还有层，根据 PA1 Page3 2a 可知，给下一个的 $\dfrac{\partial E}{\partial a}$ 是 $\dfrac{\partial E}{\partial a_{next}}w^Tg'$，即  (样本数, 下一层神经元数) x (下一层神经元数, 本层神经元数) = (样本数, 本层神经元数)，因为硬定义了 `g'=1`。

代码为：

```python
def backward(self, deltaCur, learning_rate, momentum_gamma, regularization, gradReqd=True):
    grad_activation = self.activation.backward(self.a)
    delta_next = deltaCur
    self.dw = np.dot(self.x.T, delta_next)
    delta_cur = np.dot(np.multiply(grad_activation, delta_next), self.w.T)
    if gradReqd:
        self.w -= learning_rate * self.dw
    return delta_cur[:, :-1]
```

因为这里只有输入输出，暂时不需要使用返回值，下文再说。



### Neuralnetwork

初始化：添加代码

```python
# read other configs
self.learning_rate = config['learning_rate'] # 直接是 int
self.momentum = config['momentum'] # 直接是 bool
self.momentum_gamma = config['momentum_gamma']
if not self.momentum:
    self.momentum_gamma = None
self.regularization = config['L2_penalty']
```

前向传播：targets 表示真实答案，这里一定非 None

```python
def forward(self, x, targets=None):
    self.x = x
    self.targets = targets
    for layer in self.layers:
        x = layer(x)
    self.y = x

   if targets is not None:
        return self.loss(self.y, targets), self.accuracy(self.y, targets)
    return self.y
```

其中引入了准确率计算：

```python
def accuracy(self, logits, targets):
    predictions = np.argmax(logits, axis=1)
    targets = np.argmax(targets, axis=1)
    return np.mean(predictions == targets)
```

- logits 是做了 softmax 的 `(N, 10)` 矩阵，经过求 max 得到 `(N)`；对 ont hot 逆操作得到 `(N)` 维度

交叉熵：

- 避免零除，对 $log(x)$ 转化为 $\log(x+\epsilon)$
- 因为 $y$ 是独热向量，所以 $-\sum_n\sum_{k=1}^ct^n_k\ln y^n_k$ 转化为只取 下标为 $target=\arg\max t$ 的 $\sum_n\ln(y_{target_n})$
- 题目要求要求均值，再除以样本数

```python
def loss(self, logits, targets):
    m = targets.shape[0]
    # t_k one-hot, only 1 value counts
    targets = np.argmax(targets, axis=1)
    correct_log_probs = -np.log(logits[range(m), targets] + 1e-9) # avoid zero division
    loss = np.sum(correct_log_probs)
    return loss
```

反向传播：最开始从 PA1 Page3 2a 输出层结论开始，不断倒推：

```python
def backward(self, gradReqd=True):
    # from PA1 Page3 2a, the gradient is t - y
    delta = self.y - self.targets
    for layer in reversed(self.layers):
        delta = layer.backward(
            delta, 
            self.learning_rate,
            self.momentum_gamma,
            self.regularization,
            gradReqd=gradReqd)
```

### train

读取参数，实现伪代码，实现 early stop (损失error连续多次不下降就中断)

```python
def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """

    # Read in the esssential configs
    epochs = config['epochs']
    batch_size = config['batch_size']
    early_stopping_rounds = config['early_stop_epoch']
    early_stop = config['early_stop']
    if not early_stop:
        early_stopping_rounds = float('inf')
    
    n_samples = x_train.shape[0]
    best_model = None
    best_validation_error = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(epochs):
        permutation = np.random.permutation(n_samples)
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]

        # also can use util.py's generate_minibatches
        for i in range(0, n_samples, batch_size):
            end = i + batch_size
            x_batch = x_train_shuffled[i:end]
            y_batch = y_train_shuffled[i:end]
            
            model(x_batch, y_batch) # Forward
            model.backward()

        # Early stopping check
        validation_acc, validation_error = modelTest(model, x_valid, y_valid)
        
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_model = copy.deepcopy(model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_rounds:
                break

    return best_model

#This is the test method
def modelTest(model, X_test, y_test):
    """
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    loss, acc = model(X_test, y_test)
    return acc, loss
```



### 报告

训练过程：如上代码所示。

超参数：默认。

执行代码：

```sh
python main.py --experiment test_softmax
```

绘图数据收集，修改 train 以获取数据：

```python
epochs = config['epochs']
batch_size = config['batch_size']
early_stopping_rounds = config['early_stop_epoch']
early_stop = config['early_stop']
if not early_stop:
early_stopping_rounds = float('inf')

n_samples = x_train.shape[0]
best_model = None
best_validation_error = float('inf')
early_stopping_counter = 0

train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(epochs):
permutation = np.random.permutation(n_samples)
x_train_shuffled = x_train[permutation]
y_train_shuffled = y_train[permutation]

train_loss, train_acc = [], []

# also can use util.py's generate_minibatches
for i in range(0, n_samples, batch_size):
end = i + batch_size
x_batch = x_train_shuffled[i:end]
y_batch = y_train_shuffled[i:end]

loss, acc = model(x_batch, y_batch) # Forward
train_loss.append(loss)
train_acc.append(acc)
model.backward()

train_acc = np.mean(train_acc)
train_loss = np.mean(train_loss)

# Early stopping check
validation_acc, validation_error = modelTest(model, x_valid, y_valid)

print(f'{epoch}, {validation_error:.2f}, {validation_acc:.2f}, {train_loss:.2f}, {train_acc:.2f}')
train_losses.append(train_loss)
train_accs.append(train_acc)
val_losses.append(validation_error)
val_accs.append(validation_acc)

if validation_error < best_validation_error:
best_validation_error = validation_error
best_model = copy.deepcopy(model)
early_stopping_counter = 0
else:
early_stopping_counter += 1
if early_stopping_counter >= early_stopping_rounds:
break

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
x = range(len(train_losses))

ax1.plot(x, train_losses, label='train loss')
ax1.plot(x, val_losses, label='validation loss')
ax1.set_title('Train / Validation Loss')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

ax2.plot(x, train_accs, label='train accuracy')
ax2.plot(x, val_accs, label='validation accuracy')
ax2.set_title('Train / Validation Accuracy')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()

plt.show()
```

测试集准确率：直接看 main 输出。

输出：

```
0, 0.79, 0.88, 0.79, 0.87
1, 0.80, 0.89, 0.74, 0.89
2, 0.72, 0.90, 0.68, 0.90
3, 0.79, 0.88, 0.67, 0.90
4, 0.78, 0.89, 0.65, 0.90
5, 0.78, 0.89, 0.64, 0.90
Test Accuracy: 0.8983  Test Loss: 0.7202034237572436
```



## 3. Backpropagation

### 检验隐藏层正确性

直接上 `config_5.yaml` 测试正确性。

```sh
python main.py --experiment test_gradients
```

先注释掉 `gradient.py` 的 raise 和 main 的 return 1，看看网络能不能跑。

源代码里 `backward` 的：

```python
delta_cur = np.dot(delta_next, self.w.T) * grad_activation
```

发现两个层里，这三个变量的形状分别是：

```
(128, 10) (10, 129) 1
(128, 128) (128, 785) (128, 128)
```

所以，修改为：

```python
delta_cur = np.dot(np.multiply(grad_activation, delta_next), self.w.T)
```

> 之后能正常跑代码，正确率大约 87.89%，可能是学习率过高。
>
> 学习率改成 0.01，提高到 91.19%
>
> 再跑上一问的代码，发现结果正确。

测试结束，可以恢复 return 1。



下面是一些 bugs 修复：

注意到损失函数求导结果恰为 $t-y$ 的条件是不加均值即 $E=-\sum_n\sum_{k=1}^ct^n_k\ln y^n_k$，其中 t(target) 是真实值，y 是预测值，所以修改 loss 为：

```python
loss = np.sum(correct_log_probs) 
```

实际上，$10^{-9}$ 偏移项也会影响求导。虽然此时计算已经在 $O(\epsilon^2)$ 内了，尝试过将 $10^{-9}$ 修改为 $10^{-18}$，影响不大，可以不用继续优化了。

为了让绘图 loss 一致，把均值外移，放到 train 里，修改这两行：

```python
train_loss = np.mean(train_loss) / batch_size
validation_error /=  x_valid.shape[0]
```

丢回第一问去测试，发现结果不变。再次检验了正确性。

### 数值微分

> 题给公式是中心差分；比前向/反向差分更加精确。是数值微分的一种，不是自动微分。

编写 `gradient.py` 的代码，按照公式来即可：

```python
def check_grad(model, x_train, y_train):
    """
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    epsilon = 1e-2
    model(x_train, y_train) # get backward basic
    model.backward(False) # get dw
    
    def check(layer, i, j):
        model.layers[layer].w[i, j] += epsilon
        loss1, _ = model(x_train, y_train)
        model.layers[layer].w[i, j] -= 2 * epsilon
        loss2, _ = model(x_train, y_train)
        model.layers[layer].w[i, j] += epsilon # resume
        approximation = (loss1 - loss2) / (2 * epsilon)
        gradient = model.layers[layer].dw[i, j]
        delta = abs(approximation - gradient) # expected <= O(1e-4)
        print(f'{layer} {i} {j} : {approximation:.10f} {gradient:.10f} {delta:.10f}')
    
    check(1, 128, 0) # output bias
    check(0, 784, 0) # hidden bias
    check(1, 0, 0) # hidden to output weight 1
    check(1, 3, 1) # hidden to output weight 2
    check(0, 500, 9) # input to hidden weight 1
    check(0, 103, 1) # input to hidden weight 2
```

- 注意是每次只需要更新权值的具体某一个元素，而不是对整个矩阵的所有元素都加上这个增量；也因此，这种方法算量巨大，无法用于真正的梯度下降，只能用符号解或自动微分(后者本项目不涉及)



输出结果为：

```python
1 128 0 : 1.0023628647 1.0023508568 0.0000120079
0 784 0 : -0.0136611888 -0.0175663796 0.0039051908
1 0 0 : -0.1653933018 -0.1653925813 0.0000007204
1 3 1 : 1.1063918746 1.1063926002 0.0000007255
0 500 9 : 0.0011692722 0.0016345474 0.0004652752
0 103 1 : 0.0026152970 0.0035061785 0.0008908815
```

- 注意参数选取问题：

  如果对 input to hidden，选取的 i 是诸如 i=0, i=3, 实际上就是图像里最左上角的像素，则会输出 0.0000，这是因为每张图片最左上角都是一个像素，没有任何可学习的，是无效特征。



## 4. Momentum Experiments

动量方法，设 $\gamma$ 常数，设上一次的权重梯度矩阵为 $dw'$，则 $dw=\beta dw'+(1-\beta)\triangledown$，其中 $\triangledown$ 是本来没动量算的 $dw$。其中初始 $dw'$ 是全零同形矩阵。代码参见上文。

直接运行上述代码，准确率为 91.46%，离目标很远，所以需要改进。

> 修改随机数种子，修改 early stop 轮数提高，修改学习率，修改 loss 函数 EPS 项，初始值随机值域，提前引入正则化，增减层数，更换激活函数，动态学习率，均没有提升。

> 改成 RMSProp，即 $dw'=\beta dw'+(1-\beta)\triangledown^2,dw=\dfrac{dw}{\sqrt{dw'}}$，即：[参考](https://zhuanlan.zhihu.com/p/153494778)
>
> ```python
> self.prev_dw = momentum_gamma * self.prev_dw + (1 - momentum_gamma) * np.square(self.dw)
> self.dw /= (np.sqrt(self.prev_dw) + 1e-8)
> ```
>
> 结果为 91.94%，提升约等于没有。
>
> 改成 Adam，提升也约等于没有。
>
> ```python
> # 初始化
> self.m = np.zeros_like(self.w)
> self.v = np.zeros_like(self.w)
> self.t = 0
> self.epsilon = 1e-8
> ```
>
> ```python
> self.t += 1
> beta1 = beta2 = momentum_gamma
> self.m = beta1 * self.m + (1 - beta1) * self.dw 
> self.v = beta2 * self.v + (1 - beta2) * np.square(self.dw) 
> m_hat = self.m / (1 - beta1 ** self.t)
> v_hat = self.v / (1 - beta2 ** self.t)
> self.dw =  m_hat / (np.sqrt(v_hat) + self.epsilon)
> ```

> 最终经过努力，做了一个 93.59% 的模型
>
> ```yaml
> layer_specs: [784, 256, 128, 10]
> learning_rate: 0.02
> early_stop_epoch: 10
> L2_penalty: 0.01
> ```
>
> 对 train.py，在求完 `modelTest` 后：
>
> ```python
> model.learning_rate *= 0.96
> ```
>
> 使用 Adam 方法。
>
> ```
> 0, 0.3675, 0.8959, 0.5376, 0.8337
> 1, 0.3684, 0.8972, 0.3403, 0.9036
> 2, 0.3262, 0.9062, 0.3220, 0.9080
> 3, 0.3295, 0.9102, 0.3163, 0.9100
> 4, 0.3241, 0.9082, 0.3060, 0.9137
> 5, 0.3229, 0.9086, 0.2953, 0.9149
> 6, 0.3114, 0.9131, 0.2831, 0.9190
> 7, 0.3021, 0.9167, 0.2754, 0.9213
> 8, 0.2805, 0.9231, 0.2692, 0.9232
> 9, 0.2926, 0.9177, 0.2636, 0.9244
> 10, 0.2966, 0.9165, 0.2546, 0.9273
> 11, 0.2999, 0.9130, 0.2574, 0.9259
> 12, 0.2981, 0.9181, 0.2457, 0.9287
> 13, 0.2834, 0.9188, 0.2394, 0.9301
> 14, 0.2886, 0.9202, 0.2397, 0.9301
> 15, 0.2771, 0.9245, 0.2378, 0.9299
> 16, 0.2703, 0.9261, 0.2323, 0.9325
> 17, 0.2726, 0.9242, 0.2323, 0.9326
> 18, 0.2752, 0.9227, 0.2267, 0.9336
> 19, 0.2613, 0.9256, 0.2217, 0.9348
> 20, 0.2667, 0.9252, 0.2203, 0.9357
> 21, 0.2576, 0.9275, 0.2123, 0.9381
> 22, 0.2554, 0.9271, 0.2104, 0.9386
> 23, 0.2498, 0.9290, 0.2131, 0.9375
> 24, 0.2444, 0.9283, 0.2058, 0.9394
> 25, 0.2483, 0.9304, 0.2025, 0.9403
> 26, 0.2511, 0.9296, 0.2003, 0.9413
> 27, 0.2391, 0.9324, 0.1941, 0.9430
> 28, 0.2451, 0.9291, 0.1922, 0.9437
> 29, 0.2476, 0.9286, 0.1847, 0.9441
> 30, 0.2497, 0.9273, 0.1825, 0.9470
> 31, 0.2457, 0.9303, 0.1779, 0.9476
> 32, 0.2302, 0.9347, 0.1768, 0.9470
> 33, 0.2382, 0.9326, 0.1743, 0.9475
> 34, 0.2345, 0.9337, 0.1722, 0.9481
> 35, 0.2536, 0.9283, 0.1667, 0.9508
> 36, 0.2459, 0.9325, 0.1667, 0.9515
> 37, 0.2326, 0.9347, 0.1647, 0.9502
> 38, 0.2299, 0.9347, 0.1623, 0.9516
> 39, 0.2333, 0.9348, 0.1599, 0.9513
> 40, 0.2360, 0.9369, 0.1561, 0.9526
> 41, 0.2387, 0.9355, 0.1553, 0.9538
> 42, 0.2406, 0.9329, 0.1531, 0.9530
> 43, 0.2322, 0.9334, 0.1522, 0.9535
> 44, 0.2483, 0.9282, 0.1501, 0.9549
> 45, 0.2380, 0.9337, 0.1486, 0.9549
> 46, 0.2344, 0.9348, 0.1440, 0.9559
> 47, 0.2353, 0.9378, 0.1436, 0.9560
> 48, 0.2334, 0.9362, 0.1383, 0.9574
> Test Accuracy: 0.9359  Test Loss: 2319.5645444036554
> ```



## 5. Regularization Experiments

L2 正则化：损失函数增加 $\dfrac12\lambda \sum w^2$，其导数为 $\lambda\sum w$

L1 正则化：损失函数增加 $\lambda\sum|w|$，其导数为 $\lambda\sum \mathrm{sgn}(w)$

因为要适应两种正则化，修改代码：

在 `neuralnet.py` 或其他地方增加常数或可选参数：

```python
REGTYPE = 'L1' # Regularization type L1 or L2 (str)
```

对 layer 的反向传播，修改：

```python
if regularization:
    if REGTYPE == 'L2':
        self.dw += regularization * self.w
    elif REGTYPE == 'L1':
        self.dw += regularization * np.sign(self.w)
```

添加 neural network 类的辅助函数：

```python
def penlaty_loss(self, n_samples):
    '''
    added helper functions: n_samples = total samples rather than batch  size
    '''
    loss_penalty = 0
    if self.regularization:
        if REGTYPE == 'L2':
            for layer in self.layers:
                loss_penalty += 0.5 * np.sum(layer.w ** 2)
        elif REGTYPE == 'L1':
            for layer in self.layers:
                loss_penalty += np.sum(np.abs(layer.w))
    loss_penalty *= self.regularization / n_samples
    return loss_penalty
```

- 对 mini batch SGD，每一 epoch 多个 batch，更新多次和求多次正则项，数目是 B，验证集是 n_samples，去掉这个除数，会导致：可能导致在大数据集上训练时正则化作用过强，而在小数据集上训练时正则化作用不足

loss 函数加回除以 m：

```python
loss = np.sum(correct_log_probs) / m
```

因为 model 没接收训练集大小，所以对 `train.py` 修改：

```python
train_loss, train_acc, train_penalty = [], [], []
for i in range(0, n_samples, batch_size):
    ...
    loss, acc = model(x_batch, y_batch) # Forward
    train_penalty.append(model.penlaty_loss(n_samples))
...
train_loss = np.mean(train_loss) / batch_size
train_loss += np.mean(train_penalty)
...
validation_error /= x_valid.shape[0]
validation_error += model.penlaty_loss(x_valid.shape[0])
```



做实验，增加 `config_7.yaml`，内容与 `config_5.yaml` 保持一致，后面做实验时修改 `L2_penalty` 为 0.01 或 0.0001。它具体是 L2 还是 L1 视修改上文 `REGTYPE` 定。`main.py` 对应 elif 修改为对应路径。

**切记学习率改成 0.01**，~~不然会像我一样在学习率沿用第三问的 1 让损失越来越大，然后一直没找到原因乱改了两个多小时改了无数地方。~~

实验时，修改 yaml 文件的正则化权重和上面的常量 `REGTYPE`，跑几次即可。

要实现 10% more，可以先 early stop 跑出一个轮次，记录轮数，然后不 early stop 跑固定 110% 记录轮数。



## 6. Activation Experiments  

