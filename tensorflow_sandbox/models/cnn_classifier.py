import tensorflow as tf
import numpy as np

from layers import Conv2D, FullyConnectedLayer, Flatten

class CNNClassifier():
    """
    A CNN Classifier. Accepts input dimension (N x H x W x C)
    """
    def __init__(self, height, width, num_channels, num_filters, output_size, filter_size=(5, 5), stride=(1, 1), padding="VALID", 
                    hidden_sizes=None, activation=tf.nn.relu, dropout=0.0):
        self._dropout = dropout

        self.conv_layers = []

        conv_channels = [num_channels, num_filters, 32, 64, 128, 128]

        for i in range(len(conv_channels) - 1):
            self.conv_layers.append(Conv2D(conv_channels[i], conv_channels[i + 1], filter_size, stride, padding, activation=activation, name="conv{}".format(i)))
            height = int(np.floor(1 + float(height - (filter_size[0] - 1) - 1) / float(stride[0])))
            width = int(np.floor(1 + float(width - (filter_size[1] - 1) - 1) / float(stride[1])))

        self.flatten = Flatten()

        layer_dimensions = [height * width * conv_channels[-1]]

        if hidden_sizes:
            layer_dimensions.extend(hidden_sizes)

        layer_dimensions.append(output_size)

        self.fc_layers = []

        for i in range(len(layer_dimensions) - 1):
            if i == len(layer_dimensions) - 2:
                self.fc_layers.append(FullyConnectedLayer(input_size=layer_dimensions[i], 
                                    output_size=layer_dimensions[i + 1], activation=None, name="fully_connected_{}".format(i)))
                continue

            self.fc_layers.append(FullyConnectedLayer(input_size=layer_dimensions[i], 
                                output_size=layer_dimensions[i + 1], activation=activation, name="fully_connected_{}".format(i)))

    def __call__(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            x = tf.nn.dropout(x, rate=self._dropout)

        x = self.flatten(x)

        for layer in self.fc_layers:
            x = layer(x)
            x = tf.nn.dropout(x, rate=self._dropout)

        return x