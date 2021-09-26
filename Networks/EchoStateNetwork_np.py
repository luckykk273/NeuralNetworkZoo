"""
In the ESN, the task is solved by 3 steps:
    - Provide a random RNN
    - Harvest reservoir states
    - Compute output weights

If someone is interested in more discussions, below is a simple(but easy to understand) introduction:
http://www.scholarpedia.org/article/Echo_state_network

The implementation takes the github below as reference:
https://github.com/cknd/pyESN/blob/master/pyESN.py
"""
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def _harvest(x, u, y, W, W_in, W_fb, noise, noise_size, is_teacher_forcing=True, name='tanh'):
    """
    The basic discrete-time, sigmoid-unit echo state network with N reservoir units, K inputs and L outputs
    is governed by the state update equation:
        x(n+1) = f(W*x(n) + W_in*u(n+1) + W_fb*y(n)),
        where f() is a sigmoid function (usually the logistic sigmoid or the tangent function).

    And the target is to obtain the next reservoir state x(n+1).

    Two standard ways to achieve some kind of smoothing are the following:
        - Ridge regression(also known as Tikhonov regularization):
          modify the linear regression equation for the output weights

        - State noise:
          During the state harvesting,
          instead of the equation above, use a state update which adds a noise vector ν(n) to the reservoir states:
          x(n+1) = f(W*x(n) + W_in*u(n+1) + W_fb*y(n)) + v(n)

    Both methods lead to smaller output weights.
    Here we add state noise(and in `EchoStateNetwork.py` we will implement ridge regression).

    :param x: current reservoir state
    :param u: input state
    :param y: output state
    :param W: reservoir weight matrix
    :param W_in: input weight matrix
    :param W_fb: output feedback weight matrix
    :param is_teacher_forcing: determine whether there are output-to-reservoir feedback connections or not.
           (If not, the reservoir is driven by the input u(n) only)
    :param name: determine the activation function of reservoir state

    :return: the next reservoir state
    """
    if is_teacher_forcing:
        next_state = W @ x + W_in @ u + W_fb @ y
    else:
        next_state = W @ x + W_in @ u

    if name == 'sigmoid':
        return 1. / (1. + np.exp(-next_state)) + noise * np.random.randn(noise_size)
    elif name == 'tanh':
        return np.tanh(next_state) + noise * np.random.randn(noise_size)
    else:
        return next_state + noise * np.random.randn(noise_size)


def _get_output_activation(x, name='sigmoid'):
    if name == 'identity':
        return x
    elif name == 'sigmoid':
        return 1. / (1. + np.exp(-x))
    else:
        return None


def _get_loss(y_true, y_pred, name='mse'):
    # Here we provide two loss functions: mean squared error and mean absolute error.
    if name == 'mse':
        return np.square(np.subtract(y_true, y_pred)).mean()
    elif name == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    else:
        return None


def _get_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1))


def demo():
    # Load in raw data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Split raw data into training and testing set
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels

    # Define hyperparameters
    input_num = 784  # We simply see an image as an input signal contained 784 time steps
    reservoir_num = 256
    output_num = 10
    rho_desired = 0.1  # this is the spectral radius
    sparsity = 0.9  # determined whether the neurons in the reservoir r activated.
    noise = 0.001

    # Step 1: Provide a random RNN(Initialize weights)
    print('Provide a random RNN(Initialize weights)...')
    # Input weights
    W_in = np.random.randn(reservoir_num, input_num)

    # Reservoir weights
    W = np.random.randn(reservoir_num, reservoir_num)
    mask = np.random.uniform(size=(reservoir_num, reservoir_num))
    mask[np.where(mask < sparsity)] = 0
    W *= mask  # To make the reservoir weights satisfy sparsity
    rho_random = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (rho_desired / rho_random)  # rescale the reservoir weights to satisfy the echo state property

    # Feedback(Teacher forcing) weights
    W_fb = np.random.randn(reservoir_num, output_num)

    # Step 2: Harvest reservoir states
    print('Harvest reservoir states...')
    states = np.zeros((train_imgs.shape[0], reservoir_num))  # define reservoir states
    # Note that we have 55000 states(that is, 55000 images)
    for n in range(1, train_imgs.shape[0]):
        states[n, :] = _harvest(states[n-1, :], train_imgs[n, :], train_labels[n-1, :], W, W_in, W_fb,
                                noise=noise, noise_size=reservoir_num, name='tanh')

    # Step 3: Compute output weights
    print('Compute output weights...')
    # The output is obtained from the extended system state by: y(n) = g(W_out*z(n))
    extended_states = np.hstack([states, train_imgs])

    '''
    Usually some initial portion of the states thus collected are discarded 
    to accommodate for a washout of the arbitrary (random or zero) initial reservoir state needed at time 1. 
    '''
    discarded_num = 100

    # The only weights have to learn
    W_out = np.transpose(np.linalg.pinv(extended_states[discarded_num:, :]) @ train_labels[discarded_num:, :])

    # Remember the last status
    last_state = states[-1, :]
    last_input = train_imgs[-1, :]
    last_output = train_labels[-1, :]

    # We can also initialize state(random or zero)
    # last_state = np.zeros(reservoir_num)
    # last_input = np.zeros(input_num)
    # last_output = np.zeros(output_num)

    # last_state = np.random.randn(reservoir_num)
    # last_input = np.random.randn(input_num)
    # last_output = np.random.randn(output_num)

    # Compute loss and accuracy
    print('Evaluate...')
    # Evaluate training set
    sample_num = train_imgs.shape[0]
    eva_inputs = np.vstack([last_input, train_imgs])
    eva_states = np.vstack([last_state, np.zeros((sample_num, reservoir_num))])
    eva_outputs = np.vstack([last_output, np.zeros((sample_num, output_num))])

    for n in range(sample_num):
        eva_states[n+1, :] = _harvest(eva_states[n, :], eva_inputs[n+1, :], eva_outputs[n, :], W, W_in, W_fb,
                                      noise=noise, noise_size=reservoir_num, name='tanh')

        eva_outputs[n+1, :] = W_out @ np.concatenate([eva_states[n+1, :], eva_inputs[n+1, :]])

    y_pred = _get_output_activation(eva_outputs[1:], name='sigmoid')

    train_loss = _get_loss(train_labels, y_pred, 'mse')
    train_acc = _get_accuracy(train_labels, y_pred)
    print('Train: Loss={0:f}, Accuracy={1:.4f}'.format(train_loss, train_acc))

    # Evaluate testing set
    sample_num = test_imgs.shape[0]
    eva_inputs = np.vstack([last_input, test_imgs])
    eva_states = np.vstack([last_state, np.zeros((sample_num, reservoir_num))])
    eva_outputs = np.vstack([last_output, np.zeros((sample_num, output_num))])

    for n in range(sample_num):
        eva_states[n + 1, :] = _harvest(eva_states[n, :], eva_inputs[n + 1, :], eva_outputs[n, :], W, W_in, W_fb,
                                        noise=noise, noise_size=reservoir_num, name='tanh')
        eva_outputs[n + 1, :] = W_out @ np.concatenate([eva_states[n + 1, :], eva_inputs[n + 1, :]])

    y_pred = _get_output_activation(eva_outputs[1:], name='sigmoid')
    test_loss = _get_loss(test_labels, y_pred, 'mse')
    test_acc = _get_accuracy(test_labels, y_pred)
    print('Test: Loss={0:f}, Accuracy={1:.4f}'.format(test_loss, test_acc))
