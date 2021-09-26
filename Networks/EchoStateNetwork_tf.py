"""
In the `EchoStateNetwork_np.py`, there r more discussion with mathematics and more intuitive code writing.
Here we didn't consider the code legibility.

Besides, we do a lot of effort to transform the shape of tensor between 2-D and 3-D(because of ESNCell).
In fact, we have another choice:
    Define two input placeholders which one is 2-D and the other one is 3-D,
    then we can only reshape our input data when feeding them into session.
"""
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


class ESNCell(tf.keras.layers.Layer):
    """
    This is a single RNN cell for tf.keras.layers.RNN
    """

    def __init__(self, reservoir_units, rho_desired=0.5, sparsity=0.9, leaky=0.9, **kwargs):
        def _reservoir_weights_initializer(shape, dtype=None, partition_info=None):
            W = tf.random_normal(shape, dtype=dtype)
            '''
            In numpy ver., we sample randomly from N(0, 1) to determine our mask;
            Here we randomly sample from U(0, 1).
            '''
            mask = tf.cast(tf.less_equal(tf.random_uniform(shape), sparsity), dtype)
            W = tf.multiply(W, mask)
            rho_random = tf.reduce_max(tf.abs(tf.linalg.eigvalsh(W)))
            W = W * (rho_desired / rho_random)

            return W

        self.state_size = reservoir_units
        self.leaky = leaky
        self.reservoir_initializer = _reservoir_weights_initializer

        super(ESNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_in = self.add_weight(name='input_weights', shape=(input_shape[-1], self.state_size),
                                    initializer=tf.random_normal_initializer, trainable=False)
        self.W = self.add_weight(name='reservoir_weights', shape=(self.state_size, self.state_size),
                                 initializer=self.reservoir_initializer, trainable=False)
        self.built = True

    def call(self, inputs, states):
        x_pre = states[0]  # x(n-1)
        x_n = tf.math.tanh(tf.matmul(inputs, self.W_in) + tf.matmul(x_pre, self.W))
        x_n = (1 - self.leaky) * x_pre + self.leaky * x_n

        return x_n, [x_n]


def _pinv(a, rcond=1e-15):
    """
    The implementation of pseudo inverse of matrix takes the Stack Overflow as reference:
    https://stackoverflow.com/questions/42501715/alternative-of-numpy-linalg-pinv-in-tensorflow
    """
    s, u, v = tf.svd(a)
    # Ignore singular values close to zero to prevent numerical overflow
    limit = rcond * tf.reduce_max(s)
    non_zero = tf.greater(s, limit)
    reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(tf.shape(s)))
    lhs = tf.matmul(v, tf.matrix_diag(reciprocal))

    return tf.matmul(lhs, u, transpose_b=True)


def demo():
    # Load in raw data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images[:5000]
    train_labels = mnist.train.labels[:5000]
    test_imgs = mnist.test.images[:1000]
    test_labels = mnist.test.labels[:1000]

    # Define tensorflow graph input
    # Here we define two input shapes, one for ESN cell(must be 3-D) and one for other operations(must be 2-D)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784, 1], name='inputs')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    # Define hyperparameters
    reservoir_units = 100
    discarded_num = 500

    # Step 1: Provide a random RNN(Initialize weights)
    print('Provide a random RNN(Initialize weights)...')
    # weights r initialized in ESNCell
    esn_cell = ESNCell(reservoir_units=reservoir_units, rho_desired=0.5, sparsity=0.5, leaky=0.9)

    # Step 2: Harvest reservoir states
    print('Harvest reservoir states...')
    states = tf.keras.layers.RNN(esn_cell)(inputs)  # reservoir states r harvested in esn_cell

    # Step 3: Compute output weights
    print('Compute output weights...')
    extended_states = tf.concat([states, tf.convert_to_tensor(train_imgs)], axis=1)
    # we can washout here
    W_out = tf.transpose(tf.matmul(_pinv(extended_states[discarded_num:, :]),
                                   tf.convert_to_tensor(train_labels[discarded_num:, :], dtype=tf.float32)))
    '''
    Because W_out will only train one time, 
    we use another variable to store the trained W_out so that it can be used later.
    '''
    W_out_res = tf.Variable(tf.zeros(W_out.get_shape()))

    # Remember the last status
    # Note that the last output don't need to be remembered because teacher forcing(feedback) r not used.
    last_input = tf.expand_dims(inputs[-1, :], axis=0)
    # We should use another variable to store the last state which is trained by training data.
    last_state = tf.Variable(tf.zeros(states[-1, :].get_shape()))

    eva_inputs = tf.concat([last_input, inputs], axis=0)
    # initial_state with the last state
    initial_state = tf.concat([tf.expand_dims(last_state, axis=0), states], axis=0)
    eva_states = tf.keras.layers.RNN(esn_cell)(eva_inputs, initial_state=initial_state)

    '''
    Here use a trick to compute our outputs;
    In `EchoStateNetwork_np.py`, 
    because the output weights shape doesn't fit the concatenation shape of inputs and states, 
    we should use a for loop to get evaluating output step by step.
    -------------------------------------------------------------------------------------------
    Below is an example to understand what we do:
    
    import numpy as np
    A = np.arange(6).reshape([2, 3])
    B = np.arange(6).reshape([2, 3])
    # Method 1: Use a for loop to get dot product of two arrays
    res1 = []
    for i in range(B.shape[0]):
        res1.append(A@B[i, :])
        
    res1 = np.array(res1)
    
    # Method2: Use transpose to get dot product at one time
    res2 = (A@B.T).T
    
    print(res1 == res2)  # To check the results r equal.
    '''
    concatenated = tf.concat([eva_states, tf.squeeze(eva_inputs, axis=2)], axis=1)
    eva_outputs = tf.transpose(tf.matmul(W_out_res, tf.transpose(concatenated)))
    y = tf.sigmoid(eva_outputs)[1:, :]

    # Compute loss and accuracy
    loss = tf.reduce_mean(tf.square(y_ - y))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Evaluate for training data
        trained_W_out = sess.run(W_out, feed_dict={inputs: train_imgs.reshape([-1, 784, 1])})
        sess.run(W_out_res.assign(trained_W_out, read_value=False))

        trained_states = sess.run(states, feed_dict={inputs: train_imgs.reshape([-1, 784, 1])})
        sess.run(last_state.assign(trained_states[-1, :], read_value=False))

        train_loss, train_acc = sess.run([loss, accuracy],
                                         feed_dict={inputs: train_imgs.reshape([-1, 784, 1]),
                                                    y_: train_labels})
        test_loss, test_acc = sess.run([loss, accuracy],
                                       feed_dict={inputs: test_imgs.reshape([-1, 784, 1]),
                                                  y_: test_labels})

    print('Train: Loss={0:f}, Acc={1:f}'.format(train_loss, train_acc))
    print('Test: Loss={0:f}, Acc={1:f}'.format(test_loss, test_acc))
