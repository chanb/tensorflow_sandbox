import numpy as np
import tensorflow as tf

class FullyConnectedLayer():
    def __init__(self, input_size, output_size, activation=None, name="fully_connected", 
                weight_init=None, bias_init=None):
        with tf.variable_scope(name):
            w_init = weight_init if weight_init else tf.contrib.layers.xavier_initializer()
            b_init = bias_init if bias_init else tf.zeros_initializer()

            self._W = tf.get_variable("weight", shape=[input_size, output_size], initializer=w_init, trainable=True)
            self._b = tf.get_variable("bias", shape=[output_size, ], initializer=b_init, trainable=True)
            self._activation = activation

    def __call__(self, x):
        y = tf.matmul(x, self._W) + self._b

        if self._activation:
            return self._activation(y)

        return y

    @property
    def weights(self):
        return self._W

    @property
    def bias(self):
        return self._b


class Conv2D():
    def __init__(self, num_channels, num_filters, filter_size=(3, 3), stride=(1, 1), padding="VALID", 
                activation=None, name="conv2d", weight_init=None, bias_init=None):
        with tf.variable_scope(name):
            w_init = weight_init if weight_init else tf.contrib.layers.variance_scaling_initializer()
            b_init = bias_init if bias_init else tf.zeros_initializer()

            self._W = tf.get_variable("weight", shape=[filter_size[0], filter_size[1], num_channels, num_filters], 
                                initializer=w_init, trainable=True)
            self._b = tf.get_variable("bias", shape=[num_filters], initializer=b_init, trainable=True)

            self._stride = [1, stride[0], stride[1], 1]
            self._padding = padding
            self._activation = activation
        
    def __call__(self, x):
        """
        Expect x to be shape of (N, H, W, C)
        """
        conv = tf.nn.conv2d(x, self._W, self._stride, self._padding)

        y = tf.nn.bias_add(conv, self._b)

        if self._activation:
            return self._activation(y)
        
        return y

    @property
    def weights(self):
        return self._W

    @property
    def bias(self):
        return self._b


class Flatten():
    def __call__(self, x):
        return tf.reshape(x, (-1, np.product(x.shape[1:])))