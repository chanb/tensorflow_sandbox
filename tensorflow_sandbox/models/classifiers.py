import numpy as np
import tensorflow as tf

from layers import Conv2D, Flatten, FullyConnectedLayer


class CNNClassifier():
    """
    A CNN Classifier. Accepts input dimension (N x H x W x C)
    """
    def __init__(self, height, width, input_channels, latent_size,
                 output_size, filter_size=(5, 5), stride=(1, 1),
                 filter_channels=None, hidden_sizes=None,
                 activation=tf.nn.relu, dropout=0.0):
        self._dropout = dropout

        self._conv_layers = []

        conv_channels = [input_channels]
        if filter_channels:
            conv_channels.extend(filter_channels)
        conv_channels.append(latent_size)

        for i in range(len(conv_channels) - 1):
            self._conv_layers.append(
                Conv2D(conv_channels[i], conv_channels[i + 1], filter_size,
                       stride, ctivation=activation, name="conv{}".format(i)))

            height = \
                int(np.floor(1 + float(height - (filter_size[0] - 1) - 1) /
                    float(stride[0])))
            width = \
                int(np.floor(1 + float(width - (filter_size[1] - 1) - 1) /
                    float(stride[1])))

        self.flatten = Flatten()

        layer_dimensions = [height * width * conv_channels[-1]]
        if hidden_sizes:
            layer_dimensions.extend(hidden_sizes)
        layer_dimensions.append(output_size)

        self._fc_layers = []

        for i in range(len(layer_dimensions) - 1):
            if i == len(layer_dimensions) - 2:
                self._fc_layers.append(
                    FullyConnectedLayer(input_size=layer_dimensions[i],
                                        output_size=layer_dimensions[i + 1],
                                        activation=None,
                                        name="fully_connected_{}".format(i)))
                continue

            self._fc_layers.append(
                FullyConnectedLayer(input_size=layer_dimensions[i],
                                    output_size=layer_dimensions[i + 1],
                                    activation=activation,
                                    name="fully_connected_{}".format(i)))

        self._get_parameters()

    def __call__(self, x):
        for layer in self._conv_layers:
            x = layer(x)
            x = tf.nn.dropout(x, rate=self._dropout)

        x = self.flatten(x)

        for layer in self._fc_layers:
            x = layer(x)
            x = tf.nn.dropout(x, rate=self._dropout)

        return x

    def _get_parameters(self):
        self._weights = []
        self._biases = []
        for layer in self._conv_layers:
            self._weights.append(layer.weights)
            self._biases.append(layer.bias)

        for layer in self._fc_layers:
            self._weights.append(layer.weights)
            self._biases.append(layer.bias)

    @property
    def parameters(self):
        return self._weights, self._biases


class FullyConnectedClassifier():
    def __init__(self, input_size, output_size, hidden_sizes=None,
                 activation=tf.nn.relu, dropout=0.0):
        self._dropout = dropout
        self._flatten = flatten()

        layer_dimensions = [input_size]
        if hidden_sizes:
            layer_dimensions.extend(hidden_sizes)
        layer_dimensions.append(output_size)

        self._layers = []

        for i in range(len(layer_dimensions) - 1):
            if i == len(layer_dimensions) - 2:
                self._layers.append(
                    FullyConnectedLayer(input_size=layer_dimensions[i],
                                        output_size=layer_dimensions[i + 1],
                                        activation=None,
                                        name="fully_connected_{}".format(i)))
                continue

            self._layers.append(
                FullyConnectedLayer(input_size=layer_dimensions[i],
                                    output_size=layer_dimensions[i + 1],
                                    activation=activation,
                                    name="fully_connected_{}".format(i)))

        self._get_parameters()

    def __call__(self, x):
        x = self._flatten(x)
        for layer in self._layers:
            x = layer(x)
            x = tf.nn.dropout(x, rate=self._dropout)

        return x

    def _get_parameters(self):
        self._weights = []
        self._biases = []

        for layer in self._layers:
            self._weights.append(layer.weights)
            self._biases.append(layer.bias)

    @property
    def parameters(self):
        return self._weights, self._biases
