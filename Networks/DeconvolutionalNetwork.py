"""
Deconvolutional network in fact is the inverse of convolutional network.
We take labels as input and output images.

You may find that the network cannot train well.
There r two possible reasons:
    - The input data r 10-dimensions one-hot vectors(with just a few information) from 0 to 9:
      Generally, the less information we have, the less possible we can train well.

    - Relu function is used as an activation function after dense layers:
      In many cases, relu provides a good effect on neural networks.
      But if there r too many units not be activated initially,
      then these units won't be activated anymore because of their zero gradients(called dead neurons).

      Our input data contain lots of zeros(because of the format of one-hot) and the initialization of biases r zeros.
      We can image the output of the first dense layer W*x + b(W*almost all zeros + zeros) r still almost all zeros.
      Inputting a tensor with almost all zeros to relu function will still lead to a tensor with almost all zeros.
      So that the training process won't be good at all.

      You can change relu to sigmoid to see the effect.
      It should improve a little better.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def _weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
    return tf.get_variable(name=name, initializer=tf.zeros(shape=shape))


def _deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='SAME')


def demo():
    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels

    '''
    In CNN we have implemented the concept of scope to manage our variables and names.
    In fact, if our network is not too complicated to manage, 
    we have no necessary to use tf.name_scope() and tf.variable_scope().
    '''
    # Now our input data r labels, so the input shape is (, 10).
    x = tf.placeholder(tf.float32, [None, 10], name='x')
    # Output data r images, so the output shape is (28, 28).
    y_ = tf.placeholder(tf.float32, [None, 784], name='y_')
    y_output = tf.reshape(y_, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)

    '''
    To simplify the network, we add a suppose as follow:
        - Suppose all deconvolutional layers' strides are 2(it means the output shape will be double).

    According to the suppose and input/output shape, we can precalculate the shapes in every layer inversely as follow:
            output shape: (, 28, 28, 1) 
        ->  previous convolutional layer's output shape: (, 14, 14, 16) (Suppose the channels are 16)
        ->  previous convolutional layer's output shape: (, 7, 7, 32) (Suppose the channels are 32)

    So we have to transform the input shape (, 10) to fit (, 7, 7, 32).
    Here we trickily use two dense layers:
            input shape: (, 10)
        ->  first dense layer's shape: [10, 500]
        ->  second dense layer's shape: [500, 7x7x32]

    After two dense layers, we obtain the output shape (, 7x7x32), then we can reshape it to (, 7, 7, 32)
    '''
    # Build model: DN
    W_1 = _weight_variable(name='W_1', shape=[10, 500])
    b_1 = _bias_variable(name='b_1', shape=[500])
    dense_1 = tf.nn.relu(tf.add(tf.matmul(x, W_1), b_1))  # Try to change relu to sigmoid to see the effect
    # dense_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_1), b_1))
    dense_1 = tf.nn.dropout(dense_1, keep_prob)

    W_2 = _weight_variable(name='W_2', shape=[500, 7 * 7 * 32])
    b_2 = _bias_variable(name='b_2', shape=[7 * 7 * 32])
    dense_2 = tf.nn.relu(tf.add(tf.matmul(dense_1, W_2), b_2))  # # Try to change relu to sigmoid to see the effect
    # dense_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_1), b_1))
    dense_2 = tf.nn.dropout(dense_2, keep_prob)

    dense_2_reshape = tf.reshape(dense_2, [-1, 7, 7, 32])

    W_3 = _weight_variable(name='W_3', shape=[5, 5, 16, 32])
    output_shape_1 = tf.stack([tf.shape(x)[0], 14, 14, 16])
    deconv_1 = tf.nn.relu(_deconv2d(dense_2_reshape, W_3, output_shape_1))

    W_4 = _weight_variable(name='W_4', shape=[5, 5, 1, 16])
    output_shape_2 = tf.stack([tf.shape(x)[0], 28, 28, 1])
    decoded = tf.nn.sigmoid(_deconv2d(deconv_1, W_4, output_shape_2))

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.pow(decoded - y_output, 2))  # Here we use least mean square error
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    # Initialize the variables
    init = tf.compat.v1.global_variables_initializer()

    # Training parameters
    batch_size = 64
    epochs = 50

    # Record training and testing results
    display_num = 10
    training_loss = []
    testing_loss = []

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # Shuffle the index at every beginning of epoch
            arr = np.arange(train_labels.shape[0])
            np.random.shuffle(arr)

            for index in range(0, train_labels.shape[0], batch_size):
                sess.run(optimizer, feed_dict={x: train_labels[arr[index: index + batch_size]],
                                               y_: train_imgs[arr[index: index + batch_size]],
                                               keep_prob: 0.6})

            training_loss.append(sess.run(loss, feed_dict={x: train_labels, y_: train_imgs, keep_prob: 0.6}))
            testing_loss.append(sess.run(loss, feed_dict={x: test_labels, y_: test_imgs, keep_prob: 0.6}))

            print('Epoch:{0: d}, Train loss: {1: f}, Test loss:{2: f}'.format(epoch,
                                                                              training_loss[epoch],
                                                                              testing_loss[epoch]))

        r = np.random.randint(0, test_imgs.shape[0], display_num)
        display_imgs = test_imgs[r]
        display_labels = test_labels[r]
        decoded_imgs = np.reshape(sess.run(
            decoded, feed_dict={x: display_labels, y_: display_imgs, keep_prob: 0.6}), [-1, 28, 28])

    for i in range(display_num):
        curr_img = np.reshape(display_imgs[i], (28, 28))
        ae_img = decoded_imgs[i]

        plt.subplot(2, display_num, i + 1)
        plt.imshow(curr_img, cmap=plt.get_cmap('gray'))
        plt.subplot(2, display_num, i + 1 + display_num)
        plt.imshow(ae_img, cmap=plt.get_cmap('gray'))

    plt.show()
