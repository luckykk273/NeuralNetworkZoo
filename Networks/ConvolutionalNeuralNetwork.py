"""
    Because we have already applied convolutional layer once in `AutoEncoder.py`,
    here we will try some different(not easy) ways to implement CNN.
        - Use concept of scope in Tensorflow
          (If u want to know something more, see `ExplanationForScope.txt`)
        - Implement dropout and flatten functions without tf.nn

    Note that because we use tensorflow to implement some low-level programming, the efficiency will be not so good.
    The more important is to understand the concept of scope and theory behind dropout.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data


def _weight_variable(shape=None):
    return tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(shape=None):
    return tf.get_variable(name='b', initializer=tf.zeros(shape=shape))


def _conv2d(x, filter_size, name):
    with tf.variable_scope(name):
        W = _weight_variable(filter_size)

        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))


def _max_pool2d(x, name):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def _flatten(layer):
    with tf.variable_scope('flatten'):
        # layer_shape: [num_imgs, height, weight, channels]
        layer_shape = layer.get_shape()
        # We want to count the total number of elements from axis 1 to axis 3
        num_features = layer_shape[1: 4].num_elements()

        return tf.reshape(layer, [-1, num_features])


def _dense(x, num_units, name, activation='relu'):
    with tf.variable_scope(name):
        # Dense layer only accept inputs with shape [num_imgs, num_units], so we just easily take the input shape[-1]
        W = _weight_variable(shape=[x.get_shape()[-1], num_units])
        b = _bias_variable(shape=[num_units])

        layer = tf.add(tf.matmul(x, W), b)

        if activation == 'relu':
            return tf.nn.relu(layer)
        elif activation == 'softmax':
            return tf.nn.softmax(layer)


# Here is the implementation of dropout; also u can easily use tf.nn.dropout(x, keep_prob) to fulfill
def _dropout(nodes, keep_prob):
    if keep_prob == 0:
        '''
        tf.zeros() and tf.zeros_like() return the same things: A `Tensor` with all elements set to zero.
        The difference between them is:
            - tf.zeros(): Accept a list of integers, a tuple of integers, or a 1-D `Tensor` of type `int32`.
            - tf.zeros_like(): Accept a `Tensor`.
        '''
        return tf.zeros_like(nodes)

    # Set all values smaller than keep_prob to zeros and set all values bigger than keep_prob to ones
    mask = tf.random_uniform(tf.shape(nodes)) < keep_prob
    mask = tf.cast(mask, dtype=tf.float32)

    '''
    After dropping out the neurons, we divide the keep_prob to make the mean of output keep the same.
    It's called 'inverted dropout'.
    '''
    return tf.divide(tf.multiply(mask, nodes), keep_prob)


def demo():
    # Load in raw data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels

    '''
    Build model: CNN
    Create almost everything with tf.variable_scope()
    '''
    with tf.variable_scope('CNN'):
        # Define tensorflow graph input
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='x')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

            x_input = tf.reshape(x, [-1, 28, 28, 1], name='x_input')

        # Create layers
        with tf.variable_scope('layers'):
            conv_1 = _conv2d(x_input, filter_size=[5, 5, 1, 16], name='conv_1')
            pool_1 = _max_pool2d(conv_1, name='pool_1')

            flat = _flatten(pool_1)

            dense_1 = _dense(flat, num_units=16, name='dense_1')
            dropout_1 = _dropout(dense_1, keep_prob=0.6)
            y = _dense(dropout_1, num_units=10, name='output', activation='softmax')

    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y_), logits=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

    # Metrics definition
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Training parameters
    batch_size = 16
    epochs = 100

    # Record training and testing results
    training_accuracy = []
    training_loss = []
    testing_accuracy = []

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # Shuffle the index at every beginning of epoch
            arr = np.arange(train_imgs.shape[0])
            np.random.shuffle(arr)

            for index in range(0, train_imgs.shape[0], batch_size):
                sess.run(optimizer, {x: train_imgs[arr[index: index + batch_size]],
                                     y_: train_labels[arr[index:index + batch_size]]})

            training_accuracy.append(sess.run(accuracy, feed_dict={x: train_imgs, y_: train_labels}))

            training_loss.append(sess.run(cross_entropy, {x: train_imgs, y_: train_labels}))

            # Evaluation of model at every end of epoch
            testing_accuracy.append(accuracy_score(test_labels.argmax(1),
                                                   sess.run(y, {x: test_imgs}).argmax(1)))

            print('Epoch:{0}, Train loss: {1:f} Train acc: {2:f}, Test acc:{3}'.format(epoch,
                                                                                       training_loss[epoch],
                                                                                       training_accuracy[epoch],
                                                                                       testing_accuracy[epoch]))
    iterations = list(range(epochs))
    plt.plot(iterations, training_accuracy, label='Train')
    plt.plot(iterations, testing_accuracy, label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('iterations')
    plt.show()
    print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
    print("Test Accuracy:{0:.2f}".format(testing_accuracy[-1]))



