import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def _get_activation(x, name='identity'):
    # Here we only define two activation functions: identity(linear) and sigmoid.
    if name == 'identity':
        return x
    elif name == 'sigmoid':
        return tf.nn.sigmoid(x)
    else:
        return None


def pinv(a, rcond=1e-15):
    """
    The implementation of pseudo inverse of matrix refers to the Stack Overflow:
    https://stackoverflow.com/questions/42501715/alternative-of-numpy-linalg-pinv-in-tensorflow
    """
    s, u, v = tf.svd(a)
    # Ignore singular values close to zero to prevent numerical overflow
    limit = rcond * tf.reduce_max(s)
    non_zero = tf.greater(s, limit)
    reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(tf.shape(s)))
    lhs = tf.matmul(v, tf.matrix_diag(reciprocal))

    return tf.matmul(lhs, u, transpose_b=True)


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

    # Define tensorflow graph input
    x = tf.placeholder(tf.float32, [None, input_nodes], name='x')
    y_ = tf.placeholder(tf.float32, [None, output_nodes], name='y_')
    beta = tf.placeholder(tf.float32, shape=[hidden_nodes, output_nodes], name='beta')

    # Initialize weights, biases and beta
    # Let's take another random initialization for weights and biases(different from `ExtremeLearningMachine_np.py`)
    W = tf.get_variable(
        name='W',
        shape=[input_nodes, hidden_nodes],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=False
        )
    b = tf.get_variable(
        name='b',
        shape=[hidden_nodes],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=False
        )

    # Train the model
    H = _get_activation(tf.add(tf.matmul(x, W), b), name='relu')
    '''
    Here we have to define our own function of pusedo inverse of matrix,
    because in Tensorflow 1.14 there is no function can be called.
    (In Tensorflow >= 2.0, u can simply call tf.linalg.pinv() to implement)
    '''
    H_pinv = pinv(H)
    beta_optimal = tf.matmul(H_pinv, y_)

    # Compute loss and accuracy
    y = tf.matmul(_get_activation(tf.add(tf.matmul(x, W), b), name='relu'), beta_optimal)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y_), logits=y))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        print('Train model...')
        beta_output = np.array(sess.run(beta_optimal, feed_dict={x: train_imgs, y_: train_labels}))

        print('Evaluate...')
        train_loss, train_acc = sess.run([loss, accuracy],
                                         feed_dict={x: train_imgs, y_: train_labels, beta: beta_output})
        print('Train: Loss={0:.4f}, Accuracy={1:.4f}'.format(train_loss, train_acc))

        test_loss, test_acc = sess.run([loss, accuracy],
                                       feed_dict={x: test_imgs, y_: test_labels, beta: beta_output})
        print('Test: Loss={0:.4f}, Accuracy={1:.4f}'.format(test_loss, test_acc))
