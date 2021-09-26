import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data


def demo():
    # Load in raw data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels

    # Define tensorflow graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # Build model: multi-layer perceptron
    W_1 = tf.get_variable(name='W_1', shape=[784, 392], initializer=tf.glorot_uniform_initializer())
    b_1 = tf.get_variable(name='b_1', shape=[392], initializer=tf.glorot_uniform_initializer())
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
    h_1 = tf.nn.dropout(h_1, keep_prob)

    W_2 = tf.get_variable(name='W_2', shape=[392, 196], initializer=tf.glorot_uniform_initializer())
    b_2 = tf.get_variable(name='b_2', shape=[196], initializer=tf.glorot_uniform_initializer())
    h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)
    h_2 = tf.nn.dropout(h_2, keep_prob)

    W_3 = tf.get_variable(name='W_3', shape=[196, 10], initializer=tf.glorot_uniform_initializer())
    b_3 = tf.get_variable(name='b_3', shape=[10], initializer=tf.glorot_uniform_initializer())
    y = tf.matmul(h_2, W_3) + b_3

    # Define loss and optimizer
    '''
        In softmax_cross_entropy_with_logits_v2(), BP will apply on both labels and logits, 
        so we have to stop true labels' gradient manually.
    '''
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y_), logits=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Metrics definition
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Training parameters
    batch_size = 128
    epochs = 100
    dropout_prob = 0.5

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
                sess.run(optimizer, {x: train_imgs[arr[index:index + batch_size]],
                                     y_: train_labels[arr[index:index + batch_size]],
                                     keep_prob: dropout_prob})

            training_accuracy.append(sess.run(accuracy, feed_dict={x: train_imgs, y_: train_labels, keep_prob: 1}))

            training_loss.append(sess.run(cross_entropy, {x: train_imgs, y_: train_labels, keep_prob: 1}))

            # Evaluation of model at every end of epoch
            testing_accuracy.append(accuracy_score(test_labels.argmax(1),
                                                   sess.run(y, {x: test_imgs, keep_prob: 1}).argmax(1)))

            print('Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}'.format(epoch,
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
