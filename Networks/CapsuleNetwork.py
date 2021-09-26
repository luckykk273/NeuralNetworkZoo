"""
This code is the implementation of the paper below:
http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf
"""
from tensorflow.python.keras.utils import to_categorical

import numpy as np
import tensorflow as tf


EPSILON = 1e-9


def _squash(vector, axis=-1, name=None):
    with tf.name_scope(name, default_name='squash'):
        # NOTE:
        # The shape of vector which needs to be squashed is (batch size, 1152, 8).
        # We want to compute the length of the output vector(8D) of a capsule,
        # so the axis of tf.reduce_sum() must be the axis at which the dimension is(here is -1).
        # Besides, to get 1152 lengths, we have to set keepdims=True.
        squared_norm = tf.reduce_sum(tf.square(vector), axis=axis, keepdims=True)
        scale_factor = squared_norm / (1 + squared_norm)

        # plus epsilon to avoid dividing zero(gradient will be nan)
        unit_vector = vector / tf.sqrt(squared_norm + EPSILON)

        return scale_factor * unit_vector


def _routing(u_ji, b_ij, iter_num=1):
    """
    :param u_ji: prediction vectors, shape: (batch size, 1152, 10, 16, 1)
    :param b_ij: initial logits, shape: (batch size, 1152, 10, 1, 1)
    :param iter_num: how many times to iterate routing
    :return: v_j, the outputs of digit capsule layer
    """
    # In forward, u_hat_stopped = u_hat;
    # In backward, no gradient passed back from u_hat_stopped to u_hat.
    u_ji_stopped = tf.stop_gradient(u_ji, name='stop_gradient')

    for i in range(iter_num):
        with tf.variable_scope('iter_'+str(i)):
            # The coupling coefficients between capsule i and all the capsules in the layer above sum to 1 and are
            # determined by a `routing softmax`.
            # Because we want the sum of the probabilities of 10 digit classes equal to 1,
            # we activate b_ij with softmax at axis=2(the index of digit_caps_num).
            c_ij = tf.nn.softmax(b_ij, axis=2)

            # At last iteration, use u_ji in order to receive gradients from the following graph
            if i == iter_num - 1:
                # The total input to a capsule s_j is a weighted sum
                # over the prediction vectors u_j|i from all capsule i,
                # so we multiply c_ij and u_j|i and then reduce_sum at axis=1(the index of primary_caps_num).
                # After reduce_sum at axis=1, the shape of s_j will be (batch size, 1, 10, 16, 1)
                s_j = tf.multiply(c_ij, u_ji)
                s_j = tf.reduce_sum(s_j, axis=1, keepdims=True)

                # Here because we want to calculate the length of the 16D output vectors, so the axis=-2,
                # and the shape of v_j(after squashing s_j) is still (batch size, 1, 10, 16, 1)
                v_j = _squash(s_j, axis=-2)
            elif i < iter_num - 1:
                s_j = tf.multiply(c_ij, u_ji_stopped)
                s_j = tf.reduce_sum(s_j, axis=1, keepdims=True)

                v_j = _squash(s_j, axis=-2)

                tiled_shape = tf.shape(u_ji)[1]
                v_j_tiled = tf.tile(v_j, [1, tiled_shape, 1, 1, 1])
                agreement = tf.matmul(u_ji_stopped, v_j_tiled, transpose_a=True, name='agreement')

                b_ij += agreement

    # After reduce_sum at axis=1(the index of primary_caps_num), the axis=1 is redundant.
    return tf.squeeze(v_j, axis=1)


def demo():
    with tf.variable_scope('input'):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
        y_train, y_test = to_categorical(y_train), to_categorical(y_test)

        # Define graph input
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    # Build model: CapsNet
    with tf.variable_scope('CapsNet'):
        # The first layer: convolutional layer
        with tf.variable_scope('conv1'):
            # Define parameters of the first layer
            conv1_params = {
                'filters': 256,
                'kernel_size': 9,
                'strides': 1,
                'padding': 'valid',
                'activation': tf.nn.relu
            }
            conv1 = tf.layers.conv2d(x, name='conv1', **conv1_params)

        # The second layer: PrimaryCapsules
        with tf.variable_scope('PrimaryCapsules'):
            # PrimaryCapsules is a convolutional capsule layer with 32 channels of convolutional 8D capsules,
            # each primary capsule contains 8 convolutional units with 9x9 kernel and a stride of 2.

            # Define parameters of the second layer
            primary_caps_ch = 32  # 32 channels
            # After a 9x9 kernel, stride of 2 convolution, 20x20 will become 6x6
            primary_caps_num = primary_caps_ch * 6 * 6
            primary_caps_dim = 8  # 8D capsules

            # 32 channels of convolutional 8D capsules can be represented as follow:
            #   - First add a convolutional layer with 32*8 filters: (batch size, 6, 6, 32*8)
            #   - Second reshape 32*8 filters to 32 channels of 8D capsules: (batch size, 6, 6, 32, 8)
            #     Remember to put `8` to the last index to represent a 8D vector.
            #     ((batch size, 6, 6, 8, 32) will get a 32D vector)
            conv2_params = {
                'filters': primary_caps_ch * primary_caps_dim,
                'kernel_size': 9,
                'strides': 2,
                'padding': 'valid',
                # It's not mentioned whether the activation is used in the paper.
                # In my own opinion, using activation can lead to a better effect of shrinking.
                'activation': tf.nn.relu
            }
            conv2 = tf.layers.conv2d(conv1, name='conv2', **conv2_params)

            # Here we do a trick:
            # Since this capsule layer(PrimaryCapsules) is fully connected to the next capsule layer(DigitCapsules),
            # we can simply and directly flatten the 6x6 grids.
            # In total PrimaryCapsules will have 32x6x6=1152 capsule outputs(each output is an 8D vector).
            primary_caps = tf.reshape(conv2, shape=[-1, primary_caps_num, primary_caps_dim])

            # Get capsule outputs u_i by squashing
            primary_caps_output = _squash(primary_caps)

        # The third layer: DigitCapsules
        with tf.variable_scope('DigitCapsules'):
            # DigitCapsules has one 16D capsule per digit class,
            # and each of these capsules receives input from all the capsules in primary capsules.

            # Each capsule i wants to predict the output of every capsule j:
            # The output of capsule i(u_i) will multiply a transformation matrix W_ij to get `prediction vectors` u_j|i.
            # Define parameters of the thrid layer
            digit_caps_num = 10  # 10 digit classes
            digit_caps_dim = 16  # 16D capsules

            # We want to transform a 8D vector to a 16D vector,
            # so each transformation matrix W_ij must have a shape of (16, 8).
            # There is one 16x8 transformation matrix for each pair of capsules(i, j),
            # and there r totally 1152x10 transformation matrices.
            W_init = tf.get_variable(name='W_init',
                                     shape=(1, primary_caps_num, digit_caps_num, digit_caps_dim, primary_caps_dim),
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.01)
                                     )

            # Because we have to compute the scalar product of a batch data, so we have to tile W
            batch_size = tf.shape(x)[0]
            W_tiled = tf.tile(W_init, [batch_size, 1, 1, 1, 1], name='W_tiled')

            # Now the output shape of primary capsules is (batch size, 1152, 8)
            # and the transformation matrices shape is (batch size, 1152, 10, 16, 8).
            # The transformation from a 8D vector to a 16D vector is implemented as follow:
            #   W[16x8] @ vector[8x1] = vector[16x1]
            # So we have to expand the dimension of the output of primary capsules to 8x1: (batch size, 1152, 8, 1)
            primary_caps_output = tf.expand_dims(primary_caps_output, axis=-1, name='expand_for_vector')

            # Because the 1152 primary capsules outputs have to predict for 10 digit classes(1152x10),
            # we have to expand the dimension of the output of primary capsules to 1152x10: (batch size, 1152, 10, 8, 1)
            primary_caps_output = tf.expand_dims(primary_caps_output, axis=2, name='expand_for_digits')
            primary_caps_output = tf.tile(primary_caps_output, [1, 1, digit_caps_num, 1, 1], name='tile_for_digits')

            # Now we can get the `prediction vectors` u_j|i:
            # For each pair of capsules(i, j)(1152x10), we have a 16D prediction vector.
            prediction_vector = tf.matmul(W_tiled, primary_caps_output)  # shape: (batch size, 1152, 10, 16, 1)

            # After getting the prediction vectors u_j|i, we can implement the routing algorithm
            # The initial logits b_ij are the log prior probabilities that capsule i should be coupled to capsule j,
            # so the shape of b_ij will be (batch size, 1152, 10, 1, 1).
            # But why do we need the last two dimensions of size 1?
            # The shape of b_ij equals to the shape of coupling coefficients c_ij.
            # When we multiply c_ij (batch size, 1152, 10) and u_j|i (batch size, 1152, 10, 16, 1),
            # it means we multiply 1152x10 coupling coefficients to 1152x10 16D vectors separately.
            # After multiplying 1152x10 coupling coefficients to 1152x10 16D vectors, it is still 1152x10 16D vectors,
            # so we can trickily see one coupling coefficient as a 1x1 matrix to keep the same rank.
            b_ij = tf.zeros([batch_size, primary_caps_num, digit_caps_num, 1, 1])
            digit_caps_output = _routing(prediction_vector, b_ij, iter_num=3)  # shape: (batch size, 10, 16, 1)

        # Decoder structure to reconstruct a digit from the DigitCaps layer representation.
        # The euclidean distance between the image and the output of the Sigmoid layer is minimized during training.
        # We use the true label as reconstruction target during training.

        # First, do masking:
        # During training, we mask out all but the activity vector of the correct digit capsule
        with tf.variable_scope('mask'):
            # Mask with true label:
            # Squeeze the dimension of size 1 and multiply with true label
            masked_vector = tf.multiply(tf.squeeze(digit_caps_output, axis=-1), tf.reshape(y_, (-1, 10, 1)))

            # Now because we do reduce_sum() at axis=2,
            # the shape shape of vector_length will become (batch size, 10, 1, 1)
            vector_length = tf.sqrt(tf.reduce_sum(tf.square(digit_caps_output), axis=2, keepdims=True) + EPSILON)

        # Then we use this activity vector to reconstruct the input image.
        # The output of the digit capsule is fed into a decoder consisting of 3 fully connected layers that
        # model the pixel intensities.
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(masked_vector, shape=(batch_size, digit_caps_num * digit_caps_dim))
            fc1 = tf.layers.dense(vector_j, units=512, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=1024, activation=tf.nn.relu)
            decoded = tf.layers.dense(fc2, units=784, activation=tf.nn.sigmoid)

    # Total loss is composed of two parts:
    #   - Margin loss for digit existence
    #   - Reconstruction loss
    # Define parameters of loss
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    with tf.variable_scope('loss'):
        # Margin loss:
        # To allow for multiple digits, we use a separate margin loss, L_k for each digit capsule k:
        # L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k)max(0, ||v_k|| - m-)^2
        # where Tk = 1 iff a digit of class k is present;
        #  The λ down-weighting of the loss for absent digit classes
        #  stops the initial learning from shrinking the lengths of the activity vectors of all the digit capsules.
        max_present = tf.square(tf.maximum(0., m_plus - vector_length))
        max_absent = tf.square(tf.maximum(0., vector_length - m_minus))

        # Reshape from (batch size, 10, 1, 1) to (batch size, 10)
        max_present = tf.reshape(max_present, (batch_size, -1))
        max_absent = tf.reshape(max_absent, (batch_size, -1))

        T_k = y_
        L_k = T_k * max_present + lambda_ * (1 - T_k) * max_absent

        margin_loss = tf.reduce_mean(tf.reduce_sum(L_k, axis=-1))

        # Reconstruction loss
        x_reshape = tf.reshape(x, shape=(batch_size, -1))
        reconstruction_loss = tf.reduce_mean(tf.square(decoded - x_reshape))

        # Total loss:
        # We scale down this reconstruction loss by 0.0005 so that
        # it does not dominate the margin loss during training.
        scale_factor = 0.0005
        total_loss = margin_loss + scale_factor * reconstruction_loss

    with tf.variable_scope('accuracy'):
        y = tf.argmax(tf.squeeze(vector_length), axis=-1)
        correct = tf.equal(tf.argmax(y_, axis=-1), y)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss)

    init = tf.global_variables_initializer()

    # Define training parameters
    EPOCHS = 1
    BATCH_SIZE = 128

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(EPOCHS):
            # Shuffle the index at every beginning of epoch
            arr = np.arange(x_train.shape[0])
            np.random.shuffle(arr)

            print('Training...')
            for index in range(0, x_train.shape[0], BATCH_SIZE):
                x_batch = x_train[arr[index: index + BATCH_SIZE]]
                y_batch = y_train[arr[index: index + BATCH_SIZE]]

                _, train_loss, train_acc = sess.run([optimizer, total_loss, accuracy], {x: x_batch, y_: y_batch})

                if index % 100 == 0:
                    print('Step:%d, Train loss: %f Train acc: %.2f%%' %
                          (index, train_loss, train_acc * 100))

            print('Testing...')
            # Shuffle the index at every beginning of epoch
            arr = np.arange(x_test.shape[0])
            np.random.shuffle(arr)

            for index_2 in range(0, x_test.shape[0], BATCH_SIZE):
                x_batch = x_test[arr[index_2: index_2 + BATCH_SIZE]]
                y_batch = y_test[arr[index_2: index_2 + BATCH_SIZE]]

                test_loss, test_acc = sess.run([total_loss, accuracy], {x: x_batch, y_: y_batch})
                print('Step:%d, Test loss: %f, Test acc: %.2f%%' % (index_2, test_loss, test_acc*100))
            print('===================================================================================================')
