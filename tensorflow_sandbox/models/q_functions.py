import tensorflow as tf

from tensorflow_sandbox.models.classifiers import FullyConnectedClassifier


class FullyConnectedQFunction(FullyConnectedClassifier):
    def __init__(self, input_size, output_size, hidden_sizes=None,
                 activation=tf.nn.relu, name="fc_Q"):
        super(FullyConnectedQFunction, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            name=name)
