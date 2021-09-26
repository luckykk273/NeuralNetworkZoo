"""
The main idea of DBN is to train RBM(Restricted Boltzmann Machine) layer by layer.
The first RBM's hidden units output will become the second RBM's input;
The second RBM's hidden units output will become the third RBM's input;
.
.
.

If anyone want to know how to build a DBN in a good way, please take the URL below as reference:
https://github.com/albertbup/deep-belief-network
"""
from Networks.RestrictedBoltzmannMachine_tf import BBRBM
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Load in raw data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Split raw data into training and testing set
train_imgs = mnist.train.images
train_labels = mnist.train.labels
test_imgs = mnist.test.images
test_labels = mnist.test.labels


# Here we use BBRBM to demonstrate how to reconstruct a new image with DBN:
def demo_reconstruct():
    """
    The main idea to reconstruct the image with DBN is:
    After training RBMs layer by layer,
    we inversely transform the hidden units from the last layer to the first layer to get the initial image size.
    (that is, the visible units in the first layer.)
    """
    # Determine how many units in each hidden layer
    hidden_layer_units = [[784, 256], [256, 64]]

    # We easily build a two-layer DBN to classify:
    bbrbm_list = []
    for i in range(len(hidden_layer_units)):
        # You can define your own parameters more detailed, but here we only change the number of hidden units.
        bbrbm_list.append(BBRBM(num_visible=hidden_layer_units[i][0], num_hidden=hidden_layer_units[i][1],
                                lr=0.01, momentum=0.95, use_tqdm=True))

    # Train the DBN layer by layer
    train_inputs = train_imgs
    test_inputs = test_imgs

    print('Start to train DBN...')
    for i in range(len(bbrbm_list)):
        print('RBM: ', str(i), '...')
        errs = bbrbm_list[i].fit(train_inputs, epochs=50, batch_size=128)

        # Still, we can plot the error layer by layer
        plt.plot(errs)
        plt.title('Layer: ' + str(i))
        plt.show()

        print('Transform input...')
        # Update the input(of training and testing data) to fit the next RBM's input size
        train_inputs = bbrbm_list[i].transform(train_inputs)
        test_inputs = bbrbm_list[i].transform(test_inputs)

    print('Inverse data to fit input size...')
    for i in range(1, len(bbrbm_list)+1):
        test_inputs = bbrbm_list[-i].transform_inv(test_inputs)

    display_num = 10
    r = np.random.randint(0, test_imgs.shape[0], display_num)
    display_imgs = test_imgs[r]
    decoded_imgs = np.array(test_inputs)[r]

    for i in range(display_num):
        curr_img = np.reshape(display_imgs[i, :], (28, 28))
        ae_img = np.reshape(decoded_imgs[i, :], (28, 28))

        plt.subplot(2, display_num, i + 1)
        plt.imshow(curr_img, cmap=plt.get_cmap('gray'))
        plt.subplot(2, display_num, i + 1 + display_num)
        plt.imshow(ae_img, cmap=plt.get_cmap('gray'))

    plt.show()


# Here we use BBRBM to demonstrate how to classify with DBN:
def demo_classification():
    """
    Add another one classifier to catch the output of the last RBM
    """
    # Determine how many units in each hidden layer
    hidden_layer_units = [[784, 256], [256, 64]]

    # We easily build a two-layer DBN to classify:
    bbrbm_list = []
    for i in range(len(hidden_layer_units)):
        # You can define your own parameters more detailed, but here we only change the number of hidden units.
        bbrbm_list.append(BBRBM(num_visible=hidden_layer_units[i][0], num_hidden=hidden_layer_units[i][1],
                                lr=0.01, momentum=0.95, use_tqdm=True))

    # Train the DBN layer by layer
    train_inputs = train_imgs
    test_inputs = test_imgs

    print('Start to train DBN...')
    for i in range(len(bbrbm_list)):
        print('RBM: ', str(i), '...')
        errs = bbrbm_list[i].fit(train_inputs, epochs=100, batch_size=128)

        # Still, we can plot the error layer by layer
        plt.plot(errs)
        plt.title('Layer: ' + str(i))
        plt.show()

        print('Transform input...')
        # Update the input(of training and testing data) to fit the next RBM's input size
        train_inputs = bbrbm_list[i].transform(train_inputs)
        test_inputs = bbrbm_list[i].transform(test_inputs)

    '''
    After training the DBN, we can easily add another single-layer perceptron to classify.
    Note that the implementation below is the same as 'SingleLayerPerceptron.py'.
    The only difference is:
        -The input size of the single-layer perceptron will be the number of the last RBM's hidden units
    '''
    # Define tensorflow graph input
    input_size = hidden_layer_units[-1][1]
    x = tf.placeholder(tf.float32, [None, input_size])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build model: single-layer perceptron
    W = tf.get_variable(name='W_1', shape=[input_size, 10], initializer=tf.glorot_uniform_initializer())
    b = tf.get_variable(name='b_1', shape=[10], initializer=tf.glorot_uniform_initializer())
    y = tf.matmul(x, W) + b

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

    # Record training and testing results
    training_accuracy = []
    training_loss = []
    testing_accuracy = []

    print('================================================')
    print('Start to train single-layer perceptron...')
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # Shuffle the index at every beginning of epoch
            arr = np.arange(train_inputs.shape[0])
            np.random.shuffle(arr)

            for index in range(0, train_inputs.shape[0], batch_size):
                sess.run(optimizer, {x: train_inputs[arr[index: index + batch_size]],
                                     y_: train_labels[arr[index:index + batch_size]]})

            training_accuracy.append(sess.run(accuracy, feed_dict={x: train_inputs, y_: train_labels}))

            training_loss.append(sess.run(cross_entropy, {x: train_inputs, y_: train_labels}))

            # Evaluation of model at every end of epoch
            testing_accuracy.append(accuracy_score(test_labels.argmax(1),
                                                   sess.run(y, {x: test_inputs}).argmax(1)))

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
