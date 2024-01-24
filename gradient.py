import numpy as np
from neuralnet import Neuralnetwork

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


def checkGradient(x_train, y_train, config):
    subsetSize = 10  # Feel free to change this
    sample_idx = np.random.randint(0, len(x_train), subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)