import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def demo():
    def _encoder(x_input):
        # Encoder hidden layer with sigmoid activation
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_input, _weights['encoder_h1']), _biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['encoder_h2']), _biases['encoder_b2']))

        return layer_2

    def _decoder(x_input):
        # Decoder hidden layer with sigmoid activation
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_input, _weights['decoder_h1']), _biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['decoder_h2']), _biases['decoder_b2']))

        return layer_2

    # Load in raw data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    test_imgs = mnist.test.images

    # Define tensorflow graph input
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # Define weights and biases
    _weights = {
        'encoder_h1': tf.Variable(tf.random_normal([784, 512])),
        'encoder_h2': tf.Variable(tf.random_normal([512, 256])),
        'decoder_h1': tf.Variable(tf.random_normal([256, 512])),
        'decoder_h2': tf.Variable(tf.random_normal([512, 784]))
    }

    _biases = {
        'encoder_b1': tf.Variable(tf.random_normal([512])),
        'encoder_b2': tf.Variable(tf.random_normal([256])),
        'decoder_b1': tf.Variable(tf.random_normal([512])),
        'decoder_b2': tf.Variable(tf.random_normal([784]))
    }

    # Build model: AE
    encoded = _encoder(x)
    decoded = _decoder(encoded)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.pow(x - decoded, 2))  # Here we use least mean square error
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Training parameters
    batch_size = 128
    epochs = 10

    # Record training and testing results
    training_loss = []
    testing_loss = []

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # Shuffle the index at every beginning of epoch
            arr = np.arange(train_imgs.shape[0])
            np.random.shuffle(arr)

            for index in range(0, train_imgs.shape[0], batch_size):
                sess.run(optimizer, feed_dict={x: train_imgs[arr[index: index+batch_size]]})

            training_loss.append(sess.run(loss, feed_dict={x: train_imgs}))
            testing_loss.append(sess.run(loss, feed_dict={x: test_imgs}))

            print('Epoch:{0}, Train loss: {1:.4f}, Test loss:{2:.4f}'.format(epoch,
                                                                             training_loss[epoch],
                                                                             testing_loss[epoch]))

        decoded_imgs = sess.run(decoded, feed_dict={x: test_imgs})

    display_num = 10

    for i in range(display_num):
        curr_img = np.reshape(test_imgs[i, :], (28, 28))
        ae_img = np.reshape(decoded_imgs[i, :], (28, 28))

        plt.subplot(2, display_num, i + 1)
        plt.imshow(curr_img, cmap=plt.get_cmap('gray'))
        plt.subplot(2, display_num, i + 1 + display_num)
        plt.imshow(ae_img, cmap=plt.get_cmap('gray'))

    plt.show()
