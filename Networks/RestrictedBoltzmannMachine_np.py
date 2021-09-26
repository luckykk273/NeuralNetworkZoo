"""
Our main code all come from the URL below:
https://github.com/echen/restricted-boltzmann-machines
"""
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np
import time


class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.debug_print = True

        '''
        Initialize a weight matrix of dimensions: num_visible x num_hidden,
        using Xavier Glorot uniform distribution.
        '''
        self.weights = np.asarray(np.random.RandomState(777).uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))

        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    @staticmethod
    def _logistic(x):
        return 1. / (1 + np.exp(-x))

    def train(self, data, epochs=1000, lr=0.01):
        """
        :param lr: the coefficient determined how much to update the weights
        :param epochs: the number of times to iterate
        :param data: each row is a training sample consisting of states of visible units.
        """
        print('Start training...')
        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis=1)

        # Train the RBM
        for epoch in range(epochs):
            start = time.time()
            # Here is the positive contrastive divergence(CD) phase aka reality phase
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:, 0] = 1  # Fix the bias unit.
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)

            '''
            Note that we're using the activation 'probabilities' of the hidden states, 
            not the hidden states themselves, when computing associations.
            '''
            pos_associations = np.dot(data.T, pos_hidden_probs)

            '''
            Here is the negative CD phase aka daydream phase:
            Reconstruct the visible units and sample again from the hidden units.
            '''
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)

            '''
            Note that again we're using the activation 'probabilities' when computing associations,
            not the hidden states themselves, .
            '''
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights
            self.weights += lr * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((data - neg_visible_probs) ** 2)
            end = time.time()
            if self.debug_print:
                print("Epoch %s: error is %s, usage time: %f" % (epoch, error, (end-start)))

    def _get_state(self, data, num_state, weights):
        num_examples = data.shape[0]

        '''
        Create a matrix, where each row is to be the units (plus a bias unit) sampled from a training example.
        '''
        states = np.ones(shape=(num_examples, num_state + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the units.
        activations = np.dot(data, weights)
        # Calculate the probabilities of turning the units on.
        probs = self._logistic(activations)
        # Turn the visible units on with their specified probabilities.
        states[:, :] = probs > np.random.rand(num_examples, num_state + 1)

        # Ignore the bias units.
        states = states[:, 1:]

        return states

    def run_visible(self, data):
        return self._get_state(data, self.num_hidden, self.weights)

    def run_hidden(self, data):
        return self._get_state(data, self.num_visible, self.weights.T)

    def daydream(self, num_samples):
        """
        Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        (where each step consists of updating all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step.
        Note that we only initialize the network *once*, so these samples are correlated.

        :return: each row is a sample of the visible units produced while the network was daydreaming.
        """

        # Create a matrix, where each row is to be a sample of of the visible units with an extra bias unit.
        samples = np.ones((num_samples, self.num_visible + 1))

        # Take the first sample from a uniform distribution.
        samples[0, 1:] = np.random.rand(self.num_visible)

        '''
        Start the alternating Gibbs sampling.
        Note that we keep the hidden units binary states, but leave the visible units as real probabilities.
        '''

        for i in range(1, num_samples):
            visible = samples[i - 1, :]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i, :] = visible_states

            # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:, 1:]


def demo():
    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    test_imgs = mnist.test.images

    # Train the model
    rbm = RBM(num_visible=train_imgs.shape[1], num_hidden=256)
    rbm.train(train_imgs, epochs=1000, lr=0.1)

    '''

    '''
    # Plot the result
    display_num = 10
    r = np.random.randint(0, test_imgs.shape[0], display_num)

    display_imgs = test_imgs[r]
    predicted_imgs = rbm.run_hidden(rbm.run_visible(display_imgs))

    for i in range(display_num):
        plt.subplot(2, display_num, i + 1)
        plt.imshow(np.reshape(display_imgs[i], (28, 28)), cmap=plt.get_cmap('gray'))
        plt.subplot(2, display_num, i + 1 + display_num)
        plt.imshow(np.reshape(predicted_imgs[i], (28, 28)), cmap=plt.get_cmap('gray'))

    plt.show()


