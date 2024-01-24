import numpy as np

# 定义模型参数
input_size = 10
hidden_size = 5
output_size = 2

# 创建样本数据
X = np.random.randn(100, input_size)
y = np.random.randn(100, output_size)

# 初始化权重矩阵
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# 定义正则化参数
lambda_l2 = 0.01

# 正向传播
z1 = X.dot(W1)
a1 = np.tanh(z1)
z2 = a1.dot(W2)
a2 = z2  # 简单起见，假设输出层没有激活函数

# 计算损失
loss = np.mean((a2 - y) ** 2) + 0.5 * lambda_l2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

# 反向传播
dL_da2 = 2 * (a2 - y)
dL_dW2 = a1.T.dot(dL_da2) + lambda_l2 * W2
dL_da1 = dL_da2.dot(W2.T)
dL_dz1 = dL_da1 * (1 - np.tanh(z1)**2)
dL_dW1 = X.T.dot(dL_dz1) + lambda_l2 * W1

# 更新权重
learning_rate = 0.01
W1 -= learning_rate * dL_dW1
W2 -= learning_rate * dL_dW2
