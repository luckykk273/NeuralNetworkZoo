from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import dtypes
from tqdm import tqdm

import matplotlib.cm as cm
import numpy as np


class HopfieldNetwork(object):
    def __init__(self):
        self.num_neurons = 0
        self.W = np.zeros((self.num_neurons, self.num_neurons))

        self.num_iter = 10
        self.threshold = 0
        self.sync = False

    def train_weights(self, x):
        print("Start to train weights...")
        # Define parameters
        num_x = len(x)
        self.num_neurons = x[0].shape[0]

        # Initialize weights
        self.W = np.zeros((self.num_neurons, self.num_neurons))
        rho = np.mean(x)

        # Hebb rule
        for i in tqdm(range(num_x)):
            temp = x[i] - rho
            self.W += np.outer(temp, temp)

        '''
        Make diagonal element of W become zero: 
        When W is a multi-dimensions array, diagonal function in numpy will return a 1-dimension array;
        When W is a 1-dimension array, diagonal function in numpy will return a multi-dimensions array;
        '''
        self.W -= np.diag(np.diag(self.W))
        self.W /= num_x

    def get_energy(self, state):
        return -0.5 * state @ self.W @ state + np.sum(state * self.threshold)

    def _run(self, ini_state):
        """
        There r two training ways:
            - Synchronous
            - Asynchronous
        """
        state = ini_state
        # Compute energy of initial state
        e = self.get_energy(state)

        if self.sync:
            for i in range(self.num_iter):
                # Update state
                state = np.sign(self.W @ state - self.threshold)

                # Compute the new energy of state
                e_new = self.get_energy(state)

                # Check state is converged or not
                if e == e_new:
                    return state

                # Update energy
                e = e_new

            return state

        else:
            for i in range(self.num_iter):
                for j in range(100):
                    # Randomly select a neuron
                    idx = np.random.randint(0, self.num_neurons)

                    # Update state
                    state[idx] = np.sign(self.W[idx].T @ state - self.threshold)

                # Compute the new energy of state
                e_new = self.get_energy(state)

                # Check state is converged or not
                if e == e_new:
                    return state

                # Update energy
                e = e_new

            return state

    def predict(self, x, num_iter=10, threshold=0, sync=False):
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.sync = sync

        '''
        Copy data to avoid calling by reference:
        We will modify the data input, so here we copy the input data to record the modified state.
        '''
        copy_x = np.copy(x)

        predicted = []

        for i in tqdm(range(len(x))):
            predicted.append(self._run(copy_x[i]))

        return np.array(predicted)

    def plot_weights(self):
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.show()


def demo():
    # Utils
    def _plot(test, predicted, display_num=10):
        for i in range(display_num):
            plt.subplot(2, display_num, i + 1)
            plt.imshow(np.reshape(test[i], (28, 28)))
            plt.subplot(2, display_num, i + display_num + 1)
            plt.imshow(np.reshape(predicted[i], (28, 28)))

        plt.show()

    def _preprocessing(img):
        w, h = img.shape

        # Thresholding
        thres = threshold_mean(img)
        binary = img > thres
        shift = 2 * binary - 1  # Boolean to int

        return np.reshape(shift, w*h)

    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True, dtype=dtypes.uint8)

    # Split raw data into training and testing set
    train_imgs = np.reshape(mnist.train.images, (-1, 28, 28))
    test_imgs = np.reshape(mnist.test.images, (-1, 28, 28))

    # Preprocess input data
    data = np.array([_preprocessing(img) for img in train_imgs])

    # Build model: HN
    model = HopfieldNetwork()
    model.train_weights(data)

    test = np.array([_preprocessing(img) for img in test_imgs])

    display_num = 10
    r = np.random.randint(0, test.shape[0], display_num)


    '''
    Note that there r three important things have to know:
        - Threshold is a very important parameter to generate a good image.
          It determines whether a neuron(that is an input node) is activated or not.
          
        - Suggest to set the parameter sync always equals to False.
          Because computations may oscillate if neurons are updated in parallel,
          and always converge if neurons are updated sequentially.
          
        - The update may get stuck in a local optimization,
          so approaches like simulated annealing may be needed to prevent it.
    '''
    predicted = model.predict(test[r], num_iter=10, threshold=100, sync=False)

    _plot(test[r], predicted, display_num=display_num)
    model.plot_weights()

