import copy
from neuralnet import *

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
    best_validation_acc = float('-inf')
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
        validation_acc, _ = modelTest(model, x_valid, y_valid)
        
        print(epoch, _, validation_acc)
        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            best_model = copy.deepcopy(model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_rounds:
                # print(f"Stopping early at epoch {epoch}")
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