import gradient
from constants import *
from train import *
from gradient import *
import argparse


# TODO
def main(args):

    # Read the required config
    # Create different config files for different experiments
    configFile = None  # Will contain the name of the config file to be loaded
    if (args.experiment == 'test_softmax'):  # Rubric #4: Softmax Regression
        configFile = "config_4.yaml"
    elif (args.experiment == 'test_gradients'):  # Rubric #5: Numerical Approximation of Gradients
        configFile = "config_5.yaml"
    elif (args.experiment == 'test_momentum'):  # Rubric #6: Momentum Experiments
        configFile = "config_6.yaml"
    elif (args.experiment == 'test_regularization'):  # Rubric #7: Regularization Experiments
        configFile = None  # Create a config file and change None to the config file name
    elif (args.experiment == 'test_activation'):  # Rubric #8: Activation Experiments
        configFile = None  # Create a config file and change None to the config file name


    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(path=datasetDir)

    # Load the configuration from the corresponding yaml file. Specify the file path and name
    config = util.load_config(configYamlPath + configFile)

    if(args.experiment == 'test_gradients'):
        gradient.checkGradient(x_train, y_train, config)
        return 1

    # Create a Neural Network object which will be our model
    model = Neuralnetwork(config)

    # train the model
    model = train(model, x_train, y_train, x_valid, y_valid, config)

    # test the model
    test_acc, test_loss = modelTest(model, x_test, y_test)

    #Print test accuracy and test loss
    print('Test Accuracy:', test_acc, ' Test Loss:', test_loss)


if __name__ == "__main__":

    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_momentum', help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)