import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
from datetime import datetime

from models.fully_connected_classifier import FullyConnectedClassifier
from models.cnn_classifier import CNNClassifier

data_path = "/Users/bryan/Documents/sandbox/data/mnist.npz"
debug = True

validation_ratio = 0.2
lr = 3e-4
weight_decay = 0.1
epochs = 100

def get_validation_split_indicies(data_length, val_ratio=0.2):
    validation_amount = int(data_length * val_ratio)
    p = np.random.choice(range(data_length), data_length, replace=False)
    return p[:-validation_amount], p[-validation_amount:]

# Get data from MNIST
(train_x, train_y), (test_x, test_y) = mnist.load_data(path=data_path)
train_idx, valid_idx = get_validation_split_indicies(len(train_x), validation_ratio)

validation_x = train_x[valid_idx]
validation_y = train_y[valid_idx]

train_x = train_x[train_idx]
train_y = train_y[train_idx]

train_x = np.expand_dims(train_x, axis=1).transpose(0, 2, 3, 1)

validation_x = np.expand_dims(validation_x, axis=1).transpose(0, 2, 3, 1)

height, width = train_x.shape[1:3]
hidden_output_sizes = [100]
output_size = 10

tf.reset_default_graph()

with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (len(train_x), height, width, 1))
    y = tf.placeholder(tf.int64, (len(train_y),))

# model = FullyConnectedClassifier(input_size=(height * width), output_size=10, hidden_sizes=hidden_output_sizes)
model = CNNClassifier(height=28, width=28, num_channels=1, num_filters=128, output_size=10, hidden_sizes=[128])

logits = model(x)
loss = tf.losses.sparse_softmax_cross_entropy(y, logits)
trainer = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    for epoch in range(epochs):
        _, train_loss = sess.run([trainer, loss], feed_dict={x: train_x, y: train_y})
        if (epoch + 1) % 10 == 0:
                print("Epoch: {} ==================".format(epoch + 1))
                print("Training - Loss: {}".format(train_loss))