"""
Our main code all come from the URL below:
https://github.com/meownoid/tensorfow-rbm
"""
from tensorflow.examples.tutorials.mnist import input_data

import abc
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf


def tf_xavier_init(fan_in, fan_out, const=1.0, dtype=np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))

    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)


'''
The main purpose to sample hidden units from the Bernoulli distribution is:
    To determine whether a hidden unit is to be opened or not
'''
def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))


def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)


class RBM(metaclass=abc.ABCMeta):
    def __init__(self,
                 num_visible,
                 num_hidden,
                 lr=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 error_func='mse',
                 use_tqdm=True):

        assert 0.0 <= momentum <= 1.0, 'momentum should be in range [0, 1]'

        assert error_func in {'mse', 'cosine'}, 'err_function should be either \'mse\' or \'cosine\''

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if self._use_tqdm:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = lr
        self.momentum = momentum

        self.x = tf.placeholder(tf.float32, [None, self.num_visible])
        self.y = tf.placeholder(tf.float32, [None, self.num_hidden])

        self.W = tf.Variable(tf_xavier_init(self.num_visible, self.num_hidden, const=xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.num_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.num_hidden]), dtype=tf.float32)

        self.delta_W = tf.Variable(tf_xavier_init(self.num_visible, self.num_hidden, const=xavier_const),
                                   dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.num_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.num_hidden]), dtype=tf.float32)

        self.update_weights = None
        self.update_deltas = None

        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None

        # Initialize all variables
        self._initialize_vars()

        if error_func == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.multiply(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        else:
            self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    @abc.abstractmethod
    def _initialize_vars(self):
        return NotImplemented

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_free_energy(self):
        pass

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def fit(self,
            data,
            epochs=100,
            batch_size=16,
            shuffle=True,
            verbose=True):

        assert epochs > 0
        assert batch_size > 0

        num_examples = data.shape[0]
        num_batches = num_examples // batch_size + (0 if num_examples % batch_size == 0 else 1)
        idx = np.arange(num_examples)

        if shuffle:
            data_copy = data.copy()
        else:
            data_copy = data

        errs = []

        for epoch in range(epochs):
            if verbose and not self._use_tqdm:
                print('Epoch: {: d}'.format(epoch))

            epoch_errs = np.zeros((num_batches, ))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(idx)
                data_copy = data_copy[idx]

            r_batches = range(num_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(epoch), ascii=True, file=sys.stdout)

            '''
            Here is the true training stage:
            We use partial_fit() function to add updated weights to updated deltas
            '''
            for b in r_batches:
                batch_x = data_copy[b * batch_size: (b + 1) * batch_size]
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                else:
                    print('Train error: {:.4f}'.format(err_mean))

                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

        return errs

    def get_weights(self):
        return self.sess.run(self.W), self.sess.run(self.visible_bias), self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_W': self.W,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})

        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.W.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_W': self.W,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})

        saver.restore(self.sess, filename)


class BBRBM(RBM):
    def __init__(self, *args, **kwargs):
        RBM.__init__(self, *args, **kwargs)

    def _initialize_vars(self):
        def f(x_old, x_new):
            return self.momentum * x_old + self.lr * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

        # Feed forward input data to initial hidden units' probabilities
        hidden_p = tf.nn.sigmoid(tf.add(tf.matmul(self.x, self.W), self.hidden_bias))
        # Compute the probabilities from hidden to visible
        visible_recon_p = tf.nn.sigmoid(
            tf.add(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.W)), self.visible_bias))
        # Compute the probabilities from visible to hidden
        hidden_recon_p = tf.nn.sigmoid(tf.add(tf.matmul(visible_recon_p, self.W), self.hidden_bias))

        # The positive direction from input(visible) to hidden
        positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)
        # The negative direction from hidden to output(visible, that is another input)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        # Compute energy of weights, visible biases and hidden biases
        delta_W_new = f(self.delta_W, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        # Update energy of weights, visible biases and hidden biases
        update_delta_W = self.delta_W.assign(delta_W_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        # Update weights, visible biases and hidden biases
        update_W = self.W.assign(self.W + delta_W_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        # Return the updated weights, visible biases and hidden biases with energies
        self.update_deltas = [update_delta_W, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_W, update_visible_bias, update_hidden_bias]

        '''
        API transform, transform_inv, and reconstruct call below three computing graphs to get:
            - Hidden probabilities
            - Visible probabilities
            - A fully process from visible to hidden to visible probabilities
        '''
        # Here is the process from input(visible) to hidden
        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.W) + self.hidden_bias)
        # Here is the total process from input to hidden to output(that is, another input)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.W)) + self.visible_bias)
        # Here is the process from hidden to input
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.W)) + self.visible_bias)


class GBRBM(RBM):
    def __init__(self, num_visible, num_hidden, sample_visible=False, sigma=1, **kwargs):
        self.sample_visible = sample_visible
        self.sigma = sigma

        RBM.__init__(self, num_visible, num_hidden, **kwargs)

    def _initialize_vars(self):
        def f(x_old, x_new):
            return self.momentum * x_old + self.lr * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

        hidden_p = tf.nn.sigmoid(tf.add(tf.matmul(self.x, self.W), self.hidden_bias))

        '''
        In the original source code, the contributor didn't do sigmoid here.
        The one and only reason I think is:
            If we  use sigmoid function to normalize hidden units' probabilities before, we will never get 0 mean.
        
        So here I do a little change:
            If sample_visible == True: not to use sigmoid function and sample from the normal distribution;
            Else: use sigmoid function to normalize.
        '''
        visible_recon_p = tf.add(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.W)), self.visible_bias)

        if self.sample_visible:
            visible_recon_p = sample_gaussian(visible_recon_p, self.sigma)
        else:
            visible_recon_p = tf.nn.sigmoid(visible_recon_p)

        hidden_recon_p = tf.nn.sigmoid(tf.add(tf.matmul(visible_recon_p, self.W), self.hidden_bias))

        positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        delta_W_new = f(self.delta_W, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        update_delta_W = self.delta_W.assign(delta_W_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_W = self.W.assign(self.W + delta_W_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_W, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_W, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.W) + self.hidden_bias)
        self.compute_visible = tf.matmul(self.compute_hidden, tf.transpose(self.W)) + self.visible_bias
        self.compute_visible_from_hidden = tf.matmul(self.y, tf.transpose(self.W)) + self.visible_bias


'''
There r some advices from the original contributor:
    - Use BBRBM for Bernoulli distributed data, which input values r in the interval from 0 to 1. (Ex: MNIST)
    - Use GBRBM for normal distributed data with 0 mean and sigma.
'''
def demo_BBRBM():
    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    test_imgs = mnist.test.images

    # Train the model and plot the training error
    bbrbm = BBRBM(num_visible=784, num_hidden=256, lr=0.01, momentum=0.95, use_tqdm=True)
    errs = bbrbm.fit(train_imgs, epochs=100, batch_size=128)
    plt.plot(errs)
    plt.show()

    # Plot the images
    display_num = 10
    r = np.random.randint(0, test_imgs.shape[0], display_num)
    display_imgs = test_imgs[r]

    predicted_imgs = bbrbm.reconstruct(display_imgs)

    for i in range(display_num):
        plt.subplot(2, display_num, i + 1)
        plt.imshow(np.reshape(display_imgs[i], (28, 28)), cmap=plt.get_cmap('gray'))
        plt.subplot(2, display_num, i + 1 + display_num)
        plt.imshow(np.reshape(predicted_imgs[i], (28, 28)), cmap=plt.get_cmap('gray'))

    plt.show()


def demo_GBRBM():
    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    test_imgs = mnist.test.images

    # Train the model and plot the training error
    gbrbm = GBRBM(num_visible=784, num_hidden=256, sample_visible=True, lr=0.01, momentum=0.95, use_tqdm=True)
    errs = gbrbm.fit(train_imgs, epochs=100, batch_size=128)
    plt.plot(errs)
    plt.show()

    # Plot the images
    display_num = 10
    r = np.random.randint(0, test_imgs.shape[0], display_num)
    display_imgs = test_imgs[r]

    predicted_imgs = gbrbm.reconstruct(display_imgs)

    for i in range(display_num):
        plt.subplot(2, display_num, i + 1)
        plt.imshow(np.reshape(display_imgs[i], (28, 28)), cmap=plt.get_cmap('gray'))
        plt.subplot(2, display_num, i + 1 + display_num)
        plt.imshow(np.reshape(predicted_imgs[i], (28, 28)), cmap=plt.get_cmap('gray'))

    plt.show()