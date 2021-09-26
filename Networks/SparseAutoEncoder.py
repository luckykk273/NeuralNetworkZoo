import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

from functools import reduce
from tensorflow.examples.tutorials.mnist import input_data


def demo():
    """
    The main idea of sparse autoencoder is add sparse constraint to train our network.
    Here we will add two terms to loss function:
        - Sparsity Regularization:
            We want every outputs of neurons in network become small.
            The implement is to set a value rho,
            and make average output activation value closes to it as much as possible.
            If the average output activation value is away from rho,
            the cost will become big for penalty.
            Here we use KL divergence to evaluate.
            In the example, we use average output activation to replace rho.

        - L2 Regularization
            After sparse regularization, theoretically the outputs of neurons will close to the value we set.
            Here we want to make weights become small as much as possible,
            rather than making biases become big to adjust the model.

    So the total loss is add all these three terms: E = mse + sparsity regularization + l2 regularization
    """
    def _kl_div(rho, rho_hat):
        def _sub(x):
            return tf.subtract(tf.constant(1.), x)

        def _log(x1, x2):
            return tf.multiply(x1, tf.log(tf.div(x1, x2)))

        return tf.add(_log(rho, rho_hat), _log(_sub(rho), _sub(rho_hat)))

    def _encoder(x_input):
        # Encoder hidden layer with sigmoid activation
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_input, _weights['encoder_h1']), _biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['encoder_h2']), _biases['encoder_b2']))

        return layer_1, layer_2

    def _decoder(x_input):
        # Decoder hidden layer with sigmoid activation
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_input, _weights['decoder_h1']), _biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['decoder_h2']), _biases['decoder_b2']))

        return layer_2

    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    test_imgs = mnist.test.images

    # Define tensorflow graph input
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # Define weights and biases
    _weights = {
        'encoder_h1': tf.Variable(tf.truncated_normal([784, 256], stddev=0.1)),
        'encoder_h2': tf.Variable(tf.truncated_normal([256, 32], stddev=0.1)),
        'decoder_h1': tf.Variable(tf.truncated_normal([32, 256], stddev=0.1)),
        'decoder_h2': tf.Variable(tf.truncated_normal([256, 784], stddev=0.1))
    }

    _biases = {
        'encoder_b1': tf.Variable(tf.truncated_normal([256], stddev=0.1)),
        'encoder_b2': tf.Variable(tf.truncated_normal([32], stddev=0.1)),
        'decoder_b1': tf.Variable(tf.truncated_normal([256], stddev=0.1)),
        'decoder_b2': tf.Variable(tf.truncated_normal([784], stddev=0.1))
    }

    # Build model: SAE
    encoded_l1, encoded_l2 = _encoder(x)
    decoded = _decoder(encoded_l2)

    # Define loss and optimizer
    # Note that the most difficult to deal with is adjust the hyperparameters
    alpha = 5e-6
    beta = 7.5e-5

    kl_div_loss = reduce(lambda x, y: x + y,
                         map(lambda x: tf.reduce_sum(_kl_div(0.02, tf.reduce_mean(x, 0))), [encoded_l1, encoded_l2]))

    l2_loss = reduce(lambda x, y: x + y,
                     map(lambda x: tf.nn.l2_loss(x), [_weights.get(w) for w in _weights.keys()]))

    loss = tf.reduce_mean(tf.pow(decoded - x, 2)) + alpha * l2_loss + beta * kl_div_loss
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Training parameters
    batch_size = 128
    epochs = 1000

    # Record training and testing results
    training_loss = []
    testing_loss = []

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            start = time.time()

            # Shuffle the index at every beginning of epoch
            arr = np.arange(train_imgs.shape[0])
            np.random.shuffle(arr)

            for index in range(0, train_imgs.shape[0], batch_size):
                sess.run(optimizer, feed_dict={x: train_imgs[arr[index: index+batch_size]]})

            training_loss.append(sess.run(loss, feed_dict={x: train_imgs}))
            testing_loss.append(sess.run(loss, feed_dict={x: test_imgs}))

            end = time.time()
            print('Epoch:{0: d}, Train loss: {1: f}, Test loss:{2: f}, Usage time:{3: f}'.format(epoch,
                                                                                              training_loss[epoch],
                                                                                              testing_loss[epoch],
                                                                                              (end - start)))

        decoded_imgs = sess.run(decoded, feed_dict={x: test_imgs})

    display_num = 10

    for i in range(display_num):
        curr_img = np.reshape(test_imgs[i, :], (28, 28))
        ae_img = np.reshape(decoded_imgs[i, :], (28, 28))

        plt.subplot(2, display_num, i + 1)
        plt.imshow(curr_img, cmap=plt.get_cmap('gray'))
        plt.subplot(2, display_num, i + 1 + display_num)
        plt.imshow(ae_img, cmap=plt.get_cmap('gray'))

    plt.show()
