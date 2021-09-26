"""
In `DeconvolutionalNetwork.py`, we have discussed with the problem that relu may lead to(dead neurons).
So in this case, we will try to use leaky relu to prevent from dead neurons.

In this paper, the author discussed with the effects of different activation functions.
If someone feels interested in, just have a look:
https://arxiv.org/pdf/1505.00853.pdf
"""
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


def _weight_variable(name='W', shape=None):
    return tf.get_variable(name=name, shape=shape, initializer=tf.glorot_uniform_initializer())


def _bias_variable(name='b', shape=None):
    return tf.get_variable(name=name, initializer=tf.zeros(shape=shape))


def _generator(x, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        g_W_1 = _weight_variable(name='g_W_1', shape=[100, 256])
        g_b_1 = _weight_variable(name='g_b_1', shape=[256])
        g_layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, g_W_1), g_b_1), alpha=1 / 5.5)

        g_W_2 = _weight_variable(name='g_W_2', shape=[256, 512])
        g_b_2 = _weight_variable(name='g_b_2', shape=[512])
        g_layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(g_layer_1, g_W_2), g_b_2), alpha=1 / 5.5)

        g_W_3 = _weight_variable(name='g_W_3', shape=[512, 784])
        g_b_3 = _weight_variable(name='g_b_3', shape=[784])
        '''
        `It is recommended to use hyperbolic tangent function as the output from the generator model.`
        It is mentioned in the book below:
            `Generative Adversarial Networks with Python
             Deep Learning Generative Models for Image Synthesis and Image Translation`
        This book is edited by Jason Brownlee.
        If someone is interested in, take the URL below:
        https://machinelearningmastery.com/generative_adversarial_networks/
        '''
        g_layer_3 = tf.nn.tanh(tf.add(tf.matmul(g_layer_2, g_W_3), g_b_3))

    return g_layer_3


def _discriminator(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        d_W_1 = _weight_variable(name='d_W_1', shape=[784, 512])
        d_b_1 = _weight_variable(name='d_b_1', shape=[512])
        d_layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, d_W_1), d_b_1), alpha=1 / 5.5)

        d_W_2 = _weight_variable(name='d_W_2', shape=[512, 256])
        d_b_2 = _weight_variable(name='d_b_2', shape=[256])
        d_layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(d_layer_1, d_W_2), d_b_2), alpha=1 / 5.5)

        d_W_3 = _weight_variable(name='d_W_3', shape=[256, 1])
        d_b_3 = _weight_variable(name='d_b_3', shape=[1])
        d_layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(d_layer_2, d_W_3), d_b_3))

    return d_layer_3


# Here we can sample noise from uniform distribution or normal distribution.
def _sample_from_noise(batch_size):
    return np.random.normal(size=(batch_size, 100))
    # return np.random.uniform(size=(batch_size, 100))


def demo():
    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    test_imgs = mnist.test.images

    # Define tensorflow graph input
    input_real = tf.placeholder(tf.float32, shape=[None, 784])
    input_noise = tf.placeholder(tf.float32, shape=[None, 100])

    # Build model: GAN
    # Generator
    generator = _generator(input_noise)

    # Discriminator
    discriminator_real = _discriminator(input_real)
    # Because both real and fake input r used to train the same discriminator weights, we set reuse=True.
    discriminator_fake = _discriminator(generator, reuse=True)

    '''
    Define loss:
    In GAN, we have to calculate two losses:
        - one for real image
        - one for fake input(noise)
    '''
    # epsilon is to prevent loss from being zero divided.
    epsilon = 1e-7

    # Loss for discriminator
    loss_real = tf.reduce_mean(-tf.log(discriminator_real + epsilon))
    loss_fake = tf.reduce_mean(-tf.log(1.0 - discriminator_fake + epsilon))
    discriminator_loss = loss_real + loss_fake

    '''
    Loss for generator:
    In the paper proposed GAN, 
    it is mentioned that to minimize -log(discriminator_fake) is better than to minimize -log(1 - discriminator_fake).
    -log(discriminator_fake) usually get bigger gradient in the beginning of training, 
    so that the parameters will update faster.
    Here we take the -log(discriminator_fake) to minimize.
    
    If u r interested in, just read the paper as below:
    https://arxiv.org/abs/1406.2661
    --------------------------------------------------------------------------------------------------------------------
    More discussion 1:
    There r something risky to use -log(discriminator_fake) as our generator's loss:
        - Unstable gradient:
          The fight between generator and discriminator makes the gradient unstable.
          (Image that u want to pull and push the distance between two distributions at the same time.)
          
        - Mode collapse:
          It means the generator will generate many similar images(ex: generate all digit `1`).
          To avoid the loss approaching to infinity, generator tends to be `conservative`, 
          and will generate many `safe` images.
    
    More discussion 2:
    Discriminator cannot be trained too good.
    (That means the distance between generator's output distribution and real images' distribution will be so far.)
    If the distance between two distributions becomes too far, 
    the KL divergence will be meaningless and JS divergence will be a constant.
    (That means the gradient will vanish.)
    
    If u r interested in, just read the paper as below:
    https://arxiv.org/abs/1701.04862
    '''
    generator_loss = tf.reduce_mean(-tf.log(discriminator_fake + epsilon))
    # generator_loss = tf.reduce_mean(-tf.log(1.0 - discriminator_fake + epsilon))

    '''
    Define optimizer:
    We have to update discriminator first, and then update generator.
    To fulfill it, we have to assign a list to make trainable variables update in order.
    '''
    vars_total = tf.trainable_variables()
    '''
    Because we define the variable scope for discriminator and generator, 
    we can use variable name to determine a variable is used for discriminator or generator.
    '''
    discriminator_vars = [var for var in vars_total if var.name.startswith('discriminator')]
    generator_vars = [var for var in vars_total if var.name.startswith('generator')]

    discriminator_optimizer = tf.train.AdamOptimizer(3e-4).minimize(discriminator_loss, var_list=discriminator_vars)
    generator_optimizer = tf.train.AdamOptimizer(3e-4).minimize(generator_loss, var_list=generator_vars)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Training parameters
    batch_size = 128
    epochs = 100

    # Record training and testing results
    discriminator_training_losses = []
    discriminator_testing_losses = []
    generator_losses = []

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # Shuffle the index at every beginning of epoch
            arr = np.arange(train_imgs.shape[0])
            np.random.shuffle(arr)

            for index in range(0, train_imgs.shape[0], batch_size):
                # Prepare the input data and noise data
                batch_imgs = train_imgs[arr[index: index + batch_size]]

                '''
                Because the last layer in generator is tangent function that its range is [-1, 1], 
                we must normalize(rescale) input images to range [-1, 1] to feet the output of generator.
                '''
                batch_imgs = (batch_imgs - 0.5) / 0.5
                noise = _sample_from_noise(batch_imgs.shape[0])

                # Train discriminator first
                sess.run(discriminator_optimizer, feed_dict={input_real: batch_imgs, input_noise: noise})

                # Train generator
                noise = _sample_from_noise(batch_size)
                sess.run(generator_optimizer, feed_dict={input_noise: noise})

            # Here we just want to calculate the loss(not to train), so set is_trainable_d to False.
            discriminator_training_losses.append(sess.run(discriminator_loss, feed_dict={input_real: train_imgs,
                                                                                       input_noise: noise}))
            discriminator_testing_losses.append(sess.run(discriminator_loss, feed_dict={input_real: test_imgs,
                                                                                      input_noise: noise}))

            noise = _sample_from_noise(batch_size)
            generator_losses.append(sess.run(generator_loss, feed_dict={input_noise: noise}))

            '''
            Sometimes the loss of GAN is just for reference.
            Another way to evaluate GAN is to observe whether the accuracy of the discriminator is 0.5.
            If the accuracy of the discriminator is 0.5, it means the discriminator just guesses randomly.
            '''
            print('Epoch:{0}\nDiscriminator: Train loss={1:.4f}, Test loss={2:.4f}'.format(
                epoch, discriminator_training_losses[epoch], discriminator_testing_losses[epoch]))

            print('Generator: Loss={1:.4f}'.format(epoch, generator_losses[epoch]))
            print('----------------------------------------------------')

        display_num = 10
        noise = _sample_from_noise(display_num)
        generated_imgs = np.reshape(sess.run(generator, feed_dict={input_noise: noise}), [-1, 28, 28])

    for i in range(display_num):
        plt.subplot(1, display_num, i + 1)
        plt.imshow(generated_imgs[i], cmap=plt.get_cmap('gray'))

    plt.show()






