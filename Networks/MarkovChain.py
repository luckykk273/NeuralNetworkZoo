from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import dtypes

import matplotlib.pyplot as plt
import numpy as np


def _get_patterns(x, n):
    """
    We consider N time steps as follow:
        j = 0 -> [0:N]
        j = 1 -> [1:N+1]
        .
        .
        .
    """
    patterns = {}

    for i in range(x.shape[0]):  # Control the number of images
        for j in range(784-n):  # Control the number of pixels
            '''
            Note that any mutable data structure cannot be the key of dict(Ex: list).
            So we have to change list to tuple.
            '''
            p = tuple(x[i][j:j+n])

            if p not in patterns.keys():
                patterns[p] = np.zeros(shape=256, dtype=np.int64)

            patterns[p][x[i][j+n]] += 1

    return patterns


def _num_to_prob(x):
    x_trans = {}

    for k, v in x.items():
        x_trans[k] = v / np.sum(v)

    return x_trans


def _predict_by_mc(x, prob, n):
    result = np.zeros(shape=x.shape)
    sample_space = np.arange(256)

    for i in range(x.shape[0]):
        # Set initial state to our result
        ini_state = x[i][:n]

        for index, value in np.ndenumerate(ini_state):
            result[index] = value

        temp_state = tuple(x[i][:n])

        # Use probabilities of patterns to predict the next value
        for j in range(784 - n):
            '''
            Some patterns may not in the training set.
            U can define some ways to handle it:
                - Just pass(means set to zero)
                - Sample from uniform distribution 
            '''
            if temp_state in prob.keys():
                result[i][j + n] = np.random.choice(sample_space, size=1, p=prob[temp_state])[0]
            else:
                result[i][j + n] = np.random.choice(sample_space, size=1, p=None)[0]
                # continue

            temp_state = tuple(result[i][j+1:j+(n+1)])

    return result


def demo():
    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True, dtype=dtypes.uint8)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    test_imgs = mnist.test.images[:10]

    '''
    Decide how many time steps do u want to consider:
    (Note that the bigger the N, the more time steps r considered.)
    N = 1 means: p(x_j) = p(x_j | x_j-1)
    N = 2 means: p(x_j) = p(x_j | x_j-1, x_j-2)
    .
    .
    .
    '''
    N = 3
    assert N < 784, 'Total length N must be smaller than 784!'

    print('Get patterns: ')
    # Training stage: use training set to record the probabilistic distribution of patterns.
    patterns = _get_patterns(train_imgs, N)
    print('Get patterns finished.')

    print('Transform probabilities: ')
    # Transform numbers to probabilities
    patterns_prob = _num_to_prob(patterns)
    print('Transform probabilities finished.')

    print('Get results: ')
    # Testing stage: input patterns of testing set and predict it.
    res = _predict_by_mc(test_imgs, patterns_prob, N)
    print('Get results finished.')

    display_num = 10
    display_imgs = res[: display_num]

    '''
    U may confuse why the output images look like noises.
    This is the normal phenomenon.
    Because of some reasons below:
        - We only consider the probabilities of the previous N time steps.
          (Probabilities have randomness.)
        - The probabilities don't learn to change.
    '''
    for i in range(display_num):
        plt.subplot(2, display_num, i + 1)
        plt.imshow(np.reshape(test_imgs[i], (28, 28)), cmap='gray')
        plt.subplot(2, display_num, i + 11)
        plt.imshow(np.reshape(display_imgs[i], (28, 28)), cmap='gray')

    plt.show()
