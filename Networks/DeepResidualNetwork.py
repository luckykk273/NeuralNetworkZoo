"""
In this file, we will add batch normalization.
If someone use tf.keras.layers.BatchNormalization(), there r two things u have to know:
    - After adding a batch normalization layer,
      u have to add batch normalization layer's update_ops to tf.GraphKeys.UPDATE_OPS with tf.add_to_collection()
    - Manually calling with tf.control_dependencies(update_ops) when training

This is because tf.keras.layers.BatchNormalization() api will not add this operation to update_ops automatically.

If someone is interesting in what the benefit is for adding residual network,
the original paper discussed with residual network is below:
https://arxiv.org/pdf/1512.03385v1.pdf
"""
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


layers = tf.keras.layers


# Shortcut residual block will not change the shape
def _residual_block(x, filters, kernel_size, stride, is_training=False, shortcut=False, name='residual_block'):
    _bn0 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn0')
    _bn1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn1')
    _bn2 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn2')

    if shortcut is True:
        shortcut = layers.Conv2D(filters, 1, strides=stride, name=name+'_conv0')(x)
        shortcut = _bn0(shortcut, training=is_training)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, _bn0.updates)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME', name=name + '_conv1')(x)
    x = _bn1(x, training=is_training)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, _bn1.updates)
    x = layers.Activation('relu', name=name + '_relu1')(x)

    x = layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='SAME', name=name + '_conv2')(x)
    x = _bn2(x, training=is_training)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, _bn2.updates)
    x = layers.Activation('relu', name=name + '_relu2')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])

    return layers.Activation('relu', name=name + 'relu_out')(x)


def demo():
    print('Load in images...')
    # Load in raw data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images[:10000]
    train_labels = mnist.train.labels[:10000]
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels

    # Build model: DRN
    print('Build model...')
    with tf.variable_scope('DRN'):
        # Define tensorflow graph input
        with tf.variable_scope('initialization'):
            x = tf.placeholder(tf.float32, [None, 784], name='x')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
            x_input = tf.reshape(x, [-1, 28, 28, 1], name='x_input')
            is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope('original_conv'):
            _bn0 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='bn0')

            # shape: (-1, 28, 28, 16)
            resnet = layers.Conv2D(16, (5, 5), strides=(1, 1), padding='SAME', name='conv0')(x_input)
            resnet = _bn0(resnet, training=is_training)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, _bn0.updates)  # add update_ops to GraphKeys.UPDATE_OPS
            resnet = layers.Activation('relu', name='relu0')(resnet)

        with tf.variable_scope('stack1'):
            # shape: (-1, 28, 28, 16)
            resnet = _residual_block(resnet, 16, (3, 3), (1, 1), shortcut=False, name='block1')
            resnet = _residual_block(resnet, 16, (3, 3), (1, 1), shortcut=False, name='block2')
            resnet = _residual_block(resnet, 16, (3, 3), (1, 1), shortcut=False, name='block3')

        with tf.variable_scope('stack2'):
            # shape: (-1, 14, 14, 32)
            resnet = _residual_block(resnet, 32, (3, 3), (2, 2), shortcut=True, name='block4')
            resnet = _residual_block(resnet, 32, (3, 3), (1, 1), shortcut=False, name='block5')
            resnet = _residual_block(resnet, 32, (3, 3), (1, 1), shortcut=False, name='block6')

        with tf.variable_scope('stack3'):
            # shape: (-1, 7, 7, 64)
            resnet = _residual_block(resnet, 64, (3, 3), (2, 2), shortcut=True, name='block7')
            resnet = _residual_block(resnet, 64, (3, 3), (1, 1), shortcut=False, name='block8')
            resnet = _residual_block(resnet, 64, (3, 3), (1, 1), shortcut=False, name='block9')

        with tf.variable_scope('output'):
            resnet = layers.GlobalAveragePooling2D()(resnet)
            y = layers.Dense(10, name='dense')(resnet)

    with tf.variable_scope('loss'):
        # Define loss and optimizer
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y_), logits=y))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='loss')
        with tf.control_dependencies(update_ops):  # run the update_ops
            train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy, name='train_op')

    with tf.variable_scope('accuracy'):
        # Metrics definition
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('Initialize...')
    # Initialize the variables
    init = tf.global_variables_initializer()

    # Training parameters
    batch_size = 16
    epochs = 10

    # Record training and testing results
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    testing_loss = []

    print('Train model...')
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            print('Epoch: {0:d}...'.format(epoch))
            # Shuffle the index at every beginning of epoch
            arr = np.arange(train_imgs.shape[0])
            np.random.shuffle(arr)

            for index in tqdm(range(0, train_imgs.shape[0], batch_size)):
                sess.run(train_op, {x: train_imgs[arr[index: index + batch_size]],
                                    y_: train_labels[arr[index:index + batch_size]],
                                    is_training: True})

            # Evaluation of model at every end of epoch
            training_accuracy.append(sess.run(accuracy, {x: train_imgs, y_: train_labels, is_training: True}))
            training_loss.append(sess.run(cross_entropy, {x: train_imgs, y_: train_labels, is_training: True}))
            testing_accuracy.append(sess.run(accuracy, {x: test_imgs, y_: test_labels, is_training: True}))
            testing_loss.append(sess.run(cross_entropy, {x: test_imgs, y_: test_labels, is_training: True}))

            print('Epoch:{0}, Train loss: {1:f} Train acc: {2:.4f}, Test loss: {3:f} Test acc:{4:.4f}'.format(
                epoch, training_loss[epoch], training_accuracy[epoch], testing_loss[epoch], testing_accuracy[epoch]))

    print('Result...')
    iterations = list(range(epochs))
    plt.plot(iterations, training_accuracy, label='Train')
    plt.plot(iterations, testing_accuracy, label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('iterations')
    plt.show()
    plt.plot(iterations, training_loss, label='Train')
    plt.plot(iterations, testing_loss, label='Test')
    plt.ylabel('Loss')
    plt.xlabel('iterations')
    plt.show()
    print("Train: Loss={1:.4f}, Accuracy={0:.2f}%".format(training_loss[-1], training_accuracy[-1]*100))
    print("Test: Loss={1:.4f}, Accuracy={0:.2f}%".format(testing_loss[-1], testing_accuracy[-1]*100))
