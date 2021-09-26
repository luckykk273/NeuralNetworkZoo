import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import to_categorical


def demo():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    print(x_train.shape)
    print(y_train.shape)

