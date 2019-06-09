import tensorflow as tf

from models.classifiers import CNNClassifier


class CNNQFunction(CNNClassifier):
    def __init__(self, height, width, input_channels, latent_size,
                 output_size, filter_size=(5, 5), stride=(1, 1),
                 filter_channels=None, hidden_sizes=None,
                 activation=tf.nn.relu):
        super(CNNQFunction, self).__init__(height, width, input_channels,
                                           latent_size, output_size,
                                           filter_size, stride,
                                           filter_channels, hidden_sizes,
                                           activation)
