import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data


'''
The main idea to apply LSTM on image classification is: 
A gray-scale image is a 2D array with dimension (m, n), 
which can be seen as a m steps time sequence with size n.
LSTM has some sequences process belows:
    - one to one
    - one to many
    - many to one
    - many to many(different corresponded time steps)
    - many to many(same corresponded time steps)
image classification problem can be seen as 'many to one' sequences process.
'''


def demo():
    # Initial parameters
    num_input = 28
    time_steps = 28
    n_classes = 10
    # Define ur own LSTM cells' size
    size = [64, 32]

    # Training parameters
    batch_size = 16
    epochs = 100

    # Load in raw data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images[:10000, :].reshape([-1, time_steps, num_input])
    train_labels = mnist.train.labels[:10000]
    test_imgs = mnist.test.images[:1000, :].reshape([-1, time_steps, num_input])
    test_labels = mnist.test.labels[:1000]

    # Define tensorflow graph input
    x = tf.placeholder(tf.float32, [None, time_steps, num_input])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    # Build model: Multi-layer LSTM
    # Define a LSTM cell list
    lstm_layers = [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(s) for s in size]
    # create a LSTM cell composed sequentially of a number of LSTMCells
    multi_lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_layers)

    # Every input will be called 28 times
    outputs, state = tf.compat.v1.nn.dynamic_rnn(multi_lstm_cell, x, dtype=tf.float32)

    '''
    In this case, we use another dense layer to fit the final output shape.
    (Because we only want to practice how to build a LSTM)
    Also can set the last LSTM cell units equal to number of classes,
    then we don't have to add another dense layer after LSTM.
    '''
    # The shape has to fit the number of the last LSTM cell units(that is size[-1])
    W = tf.get_variable(name='W', shape=[size[-1], n_classes], initializer=tf.glorot_uniform_initializer())
    b = tf.get_variable(name='b', shape=[n_classes], initializer=tf.glorot_uniform_initializer())
    y = tf.matmul(outputs[:, -1, :], W) + b

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
                                     y_: train_labels[arr[index:index + batch_size]]})

            training_accuracy.append(sess.run(accuracy, feed_dict={x: train_imgs, y_: train_labels}))

            training_loss.append(sess.run(cross_entropy, {x: train_imgs, y_: train_labels}))

            # Evaluation of model at every end of epoch
            testing_accuracy.append(accuracy_score(test_labels.argmax(1),
                                                   sess.run(y, {x: test_imgs}).argmax(1)))

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
