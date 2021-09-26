"""
As the reference `The Neural Network Zoo` mentioned, the DCIGNs r actually VAEs with CNNs and DNs.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def demo():
    # strides shape: (batch size, height, weight, channels)
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

    def _deconv2d(x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='SAME')

    def _encoder(x_input):
        layer_1 = tf.nn.relu(tf.add(_conv2d(x_input, _weights['W_e_conv_1']), _biases['b_e_conv_1']))
        layer_2 = tf.nn.relu(tf.add(_conv2d(layer_1, _weights['W_e_conv_2']), _biases['b_e_conv_2']))

        return layer_2

    def _decoder(x_input):
        layer_1 = tf.nn.relu(tf.add(_deconv2d(x_input, _weights['W_d_conv_1'], _output_shapes['output_shape_d_conv1']),
                                    _biases['b_d_conv_1']))
        layer_2 = tf.nn.relu(tf.add(_deconv2d(layer_1, _weights['W_d_conv_2'], _output_shapes['output_shape_d_conv2']),
                                    _biases['b_d_conv_2']))

        return layer_2

    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    test_imgs = mnist.test.images

    # Define tensorflow graph input
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_input = tf.reshape(x, [-1, 28, 28, 1])

    # Define weights, biases and output shape
    '''
    Note that strides shape is different between conv2d and conv2d_transpose:
        - conv2d: (height, weight, input channels, output channels)
        - conv2d_transpose: (height, weight, output channels, input channels)
    '''
    _weights = {
        'W_e_conv_1': tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.1)),
        'W_e_conv_2': tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1)),
        'W_d_conv_1': tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1)),
        'W_d_conv_2': tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.1))
    }

    _biases = {
        'b_e_conv_1': tf.Variable(tf.truncated_normal([16], stddev=0.1)),
        'b_e_conv_2': tf.Variable(tf.truncated_normal([32], stddev=0.1)),
        'b_d_conv_1': tf.Variable(tf.truncated_normal([16], stddev=0.1)),
        'b_d_conv_2': tf.Variable(tf.truncated_normal([1], stddev=0.1)),
    }

    _output_shapes = {
        'output_shape_d_conv1': tf.stack([tf.shape(x)[0], 14, 14, 16]),
        'output_shape_d_conv2': tf.stack([tf.shape(x)[0], 28, 28, 1])
    }

    # Build model: AE
    encoded = _encoder(x_input)
    decoded = _decoder(encoded)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.pow(decoded - x_input, 2))  # Here we use least mean square error
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    # Initialize the variables
    init = tf.compat.v1.global_variables_initializer()

    # Training parameters
    batch_size = 32
    epochs = 10

    # Record training and testing results
    display_num = 10
    training_loss = []
    testing_loss = []

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # Shuffle the index at every beginning of epoch
            arr = np.arange(train_imgs.shape[0])
            np.random.shuffle(arr)

            for index in range(0, train_imgs.shape[0], batch_size):
                sess.run(optimizer, feed_dict={x: train_imgs[arr[index: index + batch_size]]})

            training_loss.append(sess.run(loss, feed_dict={x: train_imgs}))
            testing_loss.append(sess.run(loss, feed_dict={x: test_imgs}))

            print('Epoch:{0: d}, Train loss: {1: f}, Test loss:{2: f}'.format(epoch,
                                                                              training_loss[epoch],
                                                                              testing_loss[epoch]))

        r = np.random.randint(0, test_imgs.shape[0], display_num)
        display_imgs = test_imgs[r]
        decoded_imgs = np.reshape(sess.run(decoded, feed_dict={x: display_imgs}), [-1, 28, 28])

    for i in range(display_num):
        curr_img = np.reshape(display_imgs[i], (28, 28))
        ae_img = decoded_imgs[i]

        plt.subplot(2, display_num, i + 1)
        plt.imshow(curr_img, cmap=plt.get_cmap('gray'))
        plt.subplot(2, display_num, i + 1 + display_num)
        plt.imshow(ae_img, cmap=plt.get_cmap('gray'))

    plt.show()
