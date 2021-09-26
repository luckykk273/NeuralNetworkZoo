"""
The main idea of ELM is not difficult but strong.
If someone is interested in, welcome to take the paper below as reference:
https://www.researchgate.net/publication/4116697_Extreme_learning_machine_A_new_learning_scheme_of_feedforward_neural_networks
"""
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


# If u want to add the activation after hidden layer, u can define your own activation function:
def _get_activation(x, name='identity'):
    # Here we only define two activation functions: identity(linear) and sigmoid.
    if name == 'identity':
        return x
    elif name == 'sigmoid':
        return 1. / (1. + np.exp(-x))
    else:
        return None


def _get_loss(y_true, y_pred, name='mse'):
    # Here we provide two loss functions: mean squared error and mean absolute error.
    if name == 'mse':
        return np.square(np.subtract(y_true, y_pred)).mean()
    elif name == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    else:
        return None


def _get_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1))


def demo():
    # Load in raw data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels

    # Define hyperparameters
    input_nodes = 784
    hidden_nodes = 1024
    output_nodes = 10

    # Build model: ELM
    # Initialize weights, biases and beta
    weights = np.random.uniform(-1., 1., size=(input_nodes, hidden_nodes))
    bias = np.zeros(shape=(hidden_nodes,))
    beta = np.random.uniform(-1., 1., size=(hidden_nodes, output_nodes))

    '''
    Train the model:
    ELM don't have to adjust parameters during iteration nor implement back propagation.
    Just start with random weights(and biases),
    and train it in a single step according to the least-squared fit(lowest error across all functions).
    Because weights and biases r randomly determined in the beginning, 
    the whole training process can be transformed to a linear system: H*B = T.
    We only have to calculate the pseudo inverse of H to get the optimal B: 
        B_optimal = H_pinv*T
    '''
    print('Train model...')
    # Feed data into hidden layer
    H = _get_activation(train_imgs @ weights + bias, name='sigmoid')
    # Compute the pseudo inverse of hidden layer's output
    H_pinv = np.linalg.pinv(H)
    # Update beta
    beta = H_pinv @ train_labels

    # Compute loss and accuracy
    print('Evaluate...')
    # Evaluate training set
    y_true = train_labels
    y_pred = _get_activation(train_imgs @ weights + bias, name='sigmoid') @ beta
    train_loss = _get_loss(y_true, y_pred, 'mse')
    train_acc = _get_accuracy(y_true, y_pred)
    print('Train: Loss={0:.4f}, Accuracy={1:.4f}'.format(train_loss, train_acc))

    # Evaluate testing set
    y_true = test_labels
    y_pred = _get_activation(test_imgs @ weights + bias, name='sigmoid') @ beta
    test_loss = _get_loss(y_true, y_pred, 'mse')
    test_acc = _get_accuracy(y_true, y_pred)
    print('Test: Loss={0:.4f}, Accuracy={1:.4f}'.format(test_loss, test_acc))
