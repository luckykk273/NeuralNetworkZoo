"""
This main idea of this method is:
    - Use autoencoder(or other image generating models, ex: RBM, GAN) to train

    - Train on only one class
      (in this case, we only train the images which labels are 1)

    - After training, we can classify whether an image equals to 1 or not
        - Compare testing image's loss with training loss,
          if the testing loss is much higher than training loss,
          we can say the testing image is abnormal(that is, not equals to 1)
"""
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


# strides shape: (batch size, height, weight, channels)
def _conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def _deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='SAME')


def _encoder(x_input):
    layer_1 = tf.nn.relu(tf.add(_conv2d(x_input, _weights['W_e_conv_1']), _biases['b_e_conv_1']))
    layer_2 = tf.nn.relu(tf.add(_conv2d(layer_1, _weights['W_e_conv_2']), _biases['b_e_conv_2']))

    return layer_2


def _decoder(x_input):
    layer_1 = tf.nn.relu(tf.add(_deconv2d(x_input, _weights['W_d_conv_1'], _output_shapes['output_shape_d_conv1']),
                                _biases['b_d_conv_1']))
    layer_2 = tf.nn.relu(tf.add(_deconv2d(layer_1, _weights['W_d_conv_2'], _output_shapes['output_shape_d_conv2']),
                                _biases['b_d_conv_2']))

    return layer_2


if __name__ == '__main__':
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels

    train_imgs = train_imgs[np.where(np.argmax(train_labels, axis=1) == 1)[0]]
    train_labels = train_labels[np.where(np.argmax(train_labels, axis=1) == 1)[0]]

    # Define tensorflow graph input
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_input = tf.reshape(x, [-1, 28, 28, 1])

    # Define weights, biases and output shape
    '''
    Note that strides shape is different between conv2d and conv2d_transpose:
        - conv2d: (height, weight, input channels, output channels)
        - conv2d_transpose: (height, weight, output channels, input channels)
    '''
    _weights = {
        'W_e_conv_1': tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.1)),
        'W_e_conv_2': tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1)),
        'W_d_conv_1': tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1)),
        'W_d_conv_2': tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.1))
    }

    _biases = {
        'b_e_conv_1': tf.Variable(tf.truncated_normal([16], stddev=0.1)),
        'b_e_conv_2': tf.Variable(tf.truncated_normal([32], stddev=0.1)),
        'b_d_conv_1': tf.Variable(tf.truncated_normal([16], stddev=0.1)),
        'b_d_conv_2': tf.Variable(tf.truncated_normal([1], stddev=0.1)),
    }

    _output_shapes = {
        'output_shape_d_conv1': tf.stack([tf.shape(x)[0], 14, 14, 16]),
        'output_shape_d_conv2': tf.stack([tf.shape(x)[0], 28, 28, 1])
    }

    # Build model: AE
    encoded = _encoder(x_input)
    decoded = _decoder(encoded)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.pow(decoded - x_input, 2))  # Here we use mean square error
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    # Initialize the variables
    init = tf.compat.v1.global_variables_initializer()

    # Define training parameters
    lr = 1e-3
    epochs = 1000
    batch_size = 128

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

        total_loss = sess.run(loss, feed_dict={x: train_imgs})
        print('Total loss: ', total_loss)
        print('====================================')

        correct_mean = 0

        for i in range(test_imgs.shape[0]):
            temp_loss = sess.run(loss, feed_dict={x: np.expand_dims(test_imgs[i], axis=0)})
            print('Image: %d, Loss: %f, Digit: %d' % (i, temp_loss, int(np.argmax(test_labels[i]))))

            '''
            We can calculate the loss to decide whether an image is positive example or not:
                If a loss value approximately equals to mean loss, it means this image is normal;
                But if a loss value much bigger than mean loss, it means this image is abnormal.
            
            The difficulty is to decide how much is a loss value bigger than mean loss called abnormal?
                - A simple way is to do the pre-test(Rule of thumb):
                    Ex: Use mean training loss or the highest training loss.
            '''
            if temp_loss > total_loss and np.argmax(test_labels[i]) != 1:
                correct_mean += 1
            elif temp_loss <= total_loss and np.argmax(test_labels[i]) == 1:
                correct_mean += 1

        print('Accuracy: ', correct_mean / test_labels.shape[0])
