import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data


# To calculate std of distribution of each cluster
def get_std(centroids, x, y, k):
    # Initial variables
    stds = np.zeros(k)
    cluster_size = np.zeros(k)

    for i in range(0, x.shape[0]):
        # To calculate sum of distance between every data points in each cluster and corresponded centroids
        stds[int(y[i])] += np.linalg.norm(centroids[int(y[i])] - x[i])
        cluster_size[int(y[i])] += 1

    for i in range(0, k):
        stds[i] = stds[i] / cluster_size[i]

    return stds


# Get RBF parameters
def get_rbf_param(x, k):
    # Note that n_clusters equals to the numbers of RBF neurons
    kmeans = KMeans(n_clusters=k).fit(x)
    # We can get k centroids
    centroids = np.array(kmeans.cluster_centers_)
    # Each clusters corresponds to a label
    y_labels = kmeans.labels_

    # According to the variables we get, we can calculate the std
    stds = get_std(centroids, x, y_labels, k)

    return centroids, stds


# exp{-b||x - mu||^2}
def rbf_kernel(centroid, beta, sample):
    return np.exp(-beta * np.power(np.linalg.norm(centroid - sample), 2))


def rbf_transform(x, centroids, stds):
    # Initial variables
    input_size = x.shape[0]
    k = centroids.shape[0]
    transformation = np.zeros([input_size, k])
    beta = np.zeros(k)

    # beta = 1/(sigma^2)
    for i in range(k):
        beta[i] = 1 / (2 * np.power(stds[i], 2))

    # Pass each sample to the RBF kernel
    for i in range(input_size):
        for j in range(k):
            transformation[i][j] = rbf_kernel(centroids[j], beta[j], x[i])

    return transformation


def demo():
    print('Load in data: ')
    # Load in raw data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels
    print('Data loading finished.')

    print('RBF transform: ')
    # First, we feed input vector into RBF with kmeans
    centroids, stds = get_rbf_param(train_imgs, k=392)  # k is the numbers of RBF neurons(also k clusters)
    # Implement RBF transformation
    train_imgs = rbf_transform(train_imgs, centroids, stds)
    test_imgs = rbf_transform(test_imgs, centroids, stds)
    print('RBF transformation finished.')

    print('Initialization start: ')
    # Define tensorflow graph input
    x = tf.placeholder(tf.float32, [None, 392])  # Note that here is 392(not 784) because we do RBF transform before
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build model
    W = tf.get_variable(name='W_1', shape=[392, 10], initializer=tf.glorot_uniform_initializer())
    b = tf.get_variable(name='b_1', shape=[10], initializer=tf.glorot_uniform_initializer())
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Define loss and optimizer
    '''
        In softmax_cross_entropy_with_logits_v2(), BP will apply on both labels and logits, 
        so we have to stop true labels' gradient manually.
    '''
    loss = tf.losses.mean_squared_error(y_, y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Metrics definition
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Training parameters
    batch_size = 128
    epochs = 1000

    # Record training and testing results
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    print('Initialization finished.')

    print('Training start: ')
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

            training_loss.append(sess.run(loss, {x: train_imgs, y_: train_labels}))

            # Evaluation of model at every end of epoch
            testing_accuracy.append(accuracy_score(test_labels.argmax(1),
                                                   sess.run(y, {x: test_imgs}).argmax(1)))

            print('Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}'.format(epoch,
                                                                                               training_loss[epoch],
                                                                                               training_accuracy[epoch],
                                                                                               testing_accuracy[epoch]))

    print('Training finished.')

    print('Evaluation: ')
    iterations = list(range(epochs))
    plt.plot(iterations, training_accuracy, label='Train')
    plt.plot(iterations, testing_accuracy, label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('iterations')
    plt.show()
    print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
    print("Test Accuracy:{0:.2f}".format(testing_accuracy[-1]))
