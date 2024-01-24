import numpy as np
import util

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
        x = np.clip(x, -500, 500) # avoid overflow
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

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """
        return 1  #Deliberately returning 1 for output layer case

REGTYPE = 'L2' # Regularization type L1 or L2 (str)

class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation):
        """
        Define the architecture and create placeholders.
        """
        np.random.seed(42)

        # Randomly initialize weights
        self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.x = None    # Save the input to forward in this
        self.a = None    # output without activation
        self.z = None    # Output After Activation
        self.activation = activation

        
        self.dw = 0  # Save the gradient w.r.t w in this. w already includes bias term
        self.prev_dw = np.zeros_like(self.w) # momentum
        
        # Adam parameters
        self.m = np.zeros_like(self.w) 
        self.v = np.zeros_like(self.w)  
        self.t = 0 
        self.epsilon = 1e-8 

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """
        self.x = np.hstack([x, np.ones((x.shape[0], 1))])
        self.a = np.dot(self.x, self.w)
        self.z = self.activation(self.a)
        return self.z

    def backward(self, deltaCur, learning_rate, momentum_gamma, regularization, gradReqd=True):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw
        The delta expression for any layer consists of delta and weights from the next layer and derivative of the activation function
        of weighted inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative of the activation function of the weighted
        input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it and you are free to interpret it any other way.
        Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass

        When implementing softmax regression part, just focus on implementing the single-layer case first.
        """
        grad_activation = self.activation.backward(self.a)
        delta_next = deltaCur
        self.dw = np.dot(self.x.T, delta_next)
        # self.dw = np.dot(delta_next.T, self.x).T
        delta_cur = np.dot(np.multiply(grad_activation, delta_next), self.w.T)
        
        if regularization:
            if REGTYPE == 'L2':
                self.dw += regularization * self.w
            elif REGTYPE == 'L1':
                self.dw += regularization * np.sign(self.w)
            
        if momentum_gamma:
            # raw momentum
            # self.dw = momentum_gamma * self.prev_dw + (1 - momentum_gamma) * self.dw
            # self.prev_dw = self.dw
            
            # RMSProp
            # self.prev_dw = momentum_gamma * self.prev_dw + (1 - momentum_gamma) * np.square(self.dw)
            # self.dw /= (np.sqrt(self.prev_dw) + 1e-8)
            
            # Adam
            self.t += 1
            beta1 = beta2 = momentum_gamma
            self.m = beta1 * self.m + (1 - beta1) * self.dw 
            self.v = beta2 * self.v + (1 - beta2) * np.square(self.dw) 
            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)
            self.dw =  m_hat / (np.sqrt(v_hat) + self.epsilon)

        if gradReqd:
            self.w -= learning_rate * self.dw
            # if 
        return delta_cur[:, :-1]

class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1], Activation(config['activation'])))
            elif i  == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output")))
                
        # read other configs
        self.learning_rate = config['learning_rate']
        self.momentum = config['momentum']
        self.momentum_gamma = config['momentum_gamma']
        if not self.momentum:
            self.momentum_gamma = None
        self.regularization = config['L2_penalty']

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        self.x = x
        self.targets = targets
        for layer in self.layers:
            x = layer(x)
        self.y = x
        
        if targets is not None:
            return self.loss(self.y, targets), self.accuracy(self.y, targets)
        return self.y


    def accuracy(self, logits, targets):
        '''
        added helper functions
        '''
        predictions = np.argmax(logits, axis=1)
        targets = np.argmax(targets, axis=1)
        return np.mean(predictions == targets)

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

    def loss(self, logits, targets):
        '''
        Compute the categorical cross-entropy loss and return it.
        '''
        m = targets.shape[0]
        
        # t_k one-hot, only 1 value counts
        targets = np.argmax(targets, axis=1)
        correct_log_probs = -np.log(logits[range(m), targets] + 1e-9) # avoid zero division
        loss = np.sum(correct_log_probs)
        return loss

    def backward(self, gradReqd=True):
        '''
        Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''
        # from PA1 Page3 2a, the gradient is t - y
        delta = self.y - self.targets
        for layer in reversed(self.layers):
            delta = layer.backward(
                delta, 
                self.learning_rate,
                self.momentum_gamma,
                self.regularization,
                gradReqd=gradReqd)



