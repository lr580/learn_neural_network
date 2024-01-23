import numpy as np

# 定义ReLU激活函数
def relu(a):
    return np.maximum(a, 0)

# 输入、权重和偏置
X = np.array([1, 1])
W_input_to_hidden = np.array([[1, -3], [-1, 1]])
b_hidden = np.array([-1, 0])
W_hidden_to_output = np.array([[2, 3], [3, 2]])
b_output = np.array([0, 1])
targets = np.array([3, 1])  # 真实值

# 计算隐藏层激活值
h = relu(np.dot(X, W_input_to_hidden) + b_hidden)

# 计算输出层激活值
y = relu(np.dot(h, W_hidden_to_output) + b_output)

# 计算delta
delta_output = targets - y  # 输出层delta
delta_hidden = np.dot(delta_output, W_hidden_to_output.T) * (h > 0)  # 隐藏层delta

# 学习率
learning_rate = 1

# 更新权重和偏置
# 隐藏层到输出层
W_hidden_to_output += learning_rate * np.outer(h, delta_output)
b_output += learning_rate * delta_output

# 输入层到隐藏层
W_input_to_hidden += learning_rate * np.outer(X, delta_hidden)
b_hidden += learning_rate * delta_hidden

# 输出结果
print(h, y, delta_output, delta_hidden, W_input_to_hidden, b_hidden, W_hidden_to_output, b_output)

