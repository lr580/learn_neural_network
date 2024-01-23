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

    # Read configs
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