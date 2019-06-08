import tensorflow as tf

from layers import FullyConnectedLayer, Flatten

class FullyConnectedClassifier():
    def __init__(self, input_size, output_size, hidden_sizes=None, activation=tf.nn.relu, dropout=0.0):
        self._dropout = dropout
        self.flatten = Flatten()

        layer_dimensions = [input_size]

        if hidden_sizes:
            layer_dimensions.extend(hidden_sizes)

        layer_dimensions.append(output_size)

        self.layers = []

        for i in range(len(layer_dimensions) - 1):
            if i == len(layer_dimensions) - 2:
                self.layers.append(FullyConnectedLayer(input_size=layer_dimensions[i], 
                                    output_size=layer_dimensions[i + 1], activation=None, name="fully_connected_{}".format(i)))
                continue

            self.layers.append(FullyConnectedLayer(input_size=layer_dimensions[i], 
                                output_size=layer_dimensions[i + 1], activation=activation, name="fully_connected_{}".format(i)))


    def __call__(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
            x = tf.nn.dropout(x, rate=self._dropout)

        return x