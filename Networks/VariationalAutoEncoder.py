import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


# Define customized initialization: Xavier Glorot initialization
def glorot_init(shape):
    return tf.random_normal(shape, stddev=1. / tf.sqrt(shape[0] / 2.))


def vae_loss(x_reconstruct, x_true, z_mean, z_std):
    # Reconstruction loss
    encoded_decoded_loss = - tf.reduce_sum(x_true * tf.log(1e-10 + x_reconstruct) +
                                           (1 - x_true) * tf.log(1e-10 + (1 - x_reconstruct)), 1)

    # KL-Divergence
    kl_div_loss = -0.5 * tf.reduce_sum(1 + z_std - tf.square(z_mean) - tf.exp(z_std), 1)

    return tf.reduce_mean(encoded_decoded_loss + kl_div_loss)


def demo():
    """
    The images VAE generated will be blurry(in fact not blurry, they r noisy).
    In the source paper, it mentioned that people tends to show the mean value of p(x|z): E[p(x|z)],
    rather than drawing samples from it: x ~ p(x|z).
    """
    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    test_imgs = mnist.test.images

    # Define hyperparameters for VAE
    input_shape = 784
    hidden_dim = 512
    latent_dim = 20

    # Define weights and biases
    _weights = {
        'encoder_h1': tf.Variable(glorot_init([input_shape, hidden_dim])),
        'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
        'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
        'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
        'decoder_out': tf.Variable(glorot_init([hidden_dim, input_shape]))
    }

    _biases = {
        'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
        'z_mean': tf.Variable(glorot_init([latent_dim])),
        'z_std': tf.Variable(glorot_init([latent_dim])),
        'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
        'decoder_out': tf.Variable(glorot_init([input_shape]))
    }

    # Build model: Convolutional VAE
    # Encoder
    x = tf.placeholder(tf.float32, [None, input_shape])
    encoded = tf.add(tf.matmul(x, _weights['encoder_h1']), _biases['encoder_b1'])
    encoded = tf.nn.relu(encoded)
    z_mean = tf.add(tf.matmul(encoded, _weights['z_mean']), _biases['z_mean'])
    z_std = tf.add(tf.matmul(encoded, _weights['z_std']), _biases['z_std'])

    # Sample from Gaussian distribution
    e = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
    z = z_mean + tf.exp(z_std / 2) * e

    # Decoder
    decoded = tf.add(tf.matmul(z, _weights['decoder_h1']), _biases['decoder_b1'])
    decoded = tf.nn.relu(decoded)
    decoded = tf.add(tf.matmul(decoded, _weights['decoder_out']), _biases['decoder_out'])
    decoded = tf.nn.sigmoid(decoded)

    # Define training parameters
    lr = 1e-3
    epochs = 10
    batch_size = 128

    # Define loss and optimizer
    loss = vae_loss(decoded, x, z_mean, z_std)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    # Initialize the variables
    init = tf.compat.v1.global_variables_initializer()

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
        decoded_imgs = sess.run(decoded, feed_dict={x: display_imgs})

        for i in range(display_num):
            curr_img = np.reshape(display_imgs[i], (28, 28))
            ae_img = np.reshape(decoded_imgs[i], (28, 28))

            plt.subplot(2, display_num, i + 1)
            plt.imshow(curr_img, cmap=plt.get_cmap('gray'))
            plt.subplot(2, display_num, i + 1 + display_num)
            plt.imshow(ae_img, cmap=plt.get_cmap('gray'))

        plt.show()
