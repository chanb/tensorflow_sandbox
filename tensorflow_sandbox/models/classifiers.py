import numpy as np
import tensorflow as tf

from tensorflow_sandbox.layers import Conv2D, Flatten, FullyConnectedLayer


class CNNClassifier():
    """
    A CNN Classifier. Accepts input dimension (N x H x W x C)
    """
    def __init__(self, height, width, input_channels, latent_size,
                 output_size, filter_size=(5, 5), stride=(1, 1),
                 filter_channels=None, hidden_sizes=None,
                 activation=tf.nn.relu, dropout=0.0, name="cnn_classifier"):
        self._dropout = dropout

        self._conv_layers = []

        conv_channels = [input_channels]
        if filter_channels:
            conv_channels.extend(filter_channels)
        conv_channels.append(latent_size)

        for i in range(len(conv_channels) - 1):
            self._conv_layers.append(
                Conv2D(conv_channels[i], conv_channels[i + 1], filter_size,
                       stride, activation=activation, name="{}_conv_{}"
                       .format(name, i)))

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

        for i in range(len(layer_dimensions) - 2):
            self._layers.append(
                FullyConnectedLayer(input_size=layer_dimensions[i],
                                    output_size=layer_dimensions[i + 1],
                                    activation=activation,
                                    name="{}_fully_connected_{}"
                                    .format(name, i)))

        self._layers.append(
            FullyConnectedLayer(input_size=layer_dimensions[-2],
                                output_size=layer_dimensions[-1],
                                activation=None,
                                name="{}_fully_connected_{}"
                                .format(name, len(layer_dimensions) - 2)))

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
        self._weights = dict()
        self._biases = dict()

        for i, layer in enumerate(self._conv_layers):
            self._weights['conv_W_{}'.format(i)] = layer.weights
            self._biases['conv_b_{}'.format(i)] = layer.bias

        for i, layer in enumerate(self._conv_layers):
            self._weights['fc_W_{}'.format(i)] = layer.weights
            self._biases['fc_b_{}'.format(i)] = layer.bias

        self._parameters = {**self._weights, **self._biases}

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def parameters(self):
        return self._parameters


class FullyConnectedClassifier():
    def __init__(self, input_size, output_size, hidden_sizes=None,
                 activation=tf.nn.relu, dropout=0.0, name="fc_classifier"):
        self._dropout = dropout
        self._flatten = Flatten()

        layer_dimensions = [input_size]
        if hidden_sizes:
            layer_dimensions.extend(hidden_sizes)
        layer_dimensions.append(output_size)

        self._layers = []

        for i in range(len(layer_dimensions) - 2):
            self._layers.append(
                FullyConnectedLayer(input_size=layer_dimensions[i],
                                    output_size=layer_dimensions[i + 1],
                                    activation=activation,
                                    name="{}_fully_connected_{}"
                                    .format(name, i)))

        self._layers.append(
            FullyConnectedLayer(input_size=layer_dimensions[-2],
                                output_size=layer_dimensions[-1],
                                activation=None,
                                name="{}_fully_connected_{}"
                                .format(name, len(layer_dimensions) - 2)))

        self._get_parameters()

    def __call__(self, x):
        x = self._flatten(x)
        for layer in self._layers:
            x = layer(x)
            x = tf.nn.dropout(x, rate=self._dropout)

        return x

    def _get_parameters(self):
        self._weights = dict()
        self._biases = dict()

        for i, layer in enumerate(self._layers):
            self._weights['fc_W_{}'.format(i)] = layer.weights
            self._biases['fc_b_{}'.format(i)] = layer.bias

        self._parameters = {**self._weights, **self._biases}

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def parameters(self):
        return self._parameters
