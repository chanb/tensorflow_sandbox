import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
from datetime import datetime

from models.fully_connected_classifier import FullyConnectedClassifier
from models.cnn_classifier import CNNClassifier

def get_validation_split_indicies(data_length, val_ratio=0.2):
    validation_amount = int(data_length * val_ratio)
    p = np.random.choice(range(data_length), data_length, replace=False)
    return p[:-validation_amount], p[-validation_amount:]

def get_mnist_data(data_path, validation_ratio):
    # Get data from MNIST
    (train_x, train_y), (test_x, test_y) = mnist.load_data(path=data_path)
    train_idx, valid_idx = get_validation_split_indicies(len(train_x), validation_ratio)

    validation_x = train_x[valid_idx]
    validation_y = train_y[valid_idx]

    train_x = train_x[train_idx]
    train_y = train_y[train_idx]

    # Resize the input
    train_x = np.expand_dims(train_x, axis=1).transpose(0, 2, 3, 1)
    validation_x = np.expand_dims(validation_x, axis=1).transpose(0, 2, 3, 1)

    return train_x, train_y, validation_x, validation_y, test_x, test_y

def train(data_path, validation_ratio, lr, epochs, dropout, l2_coef, model_path, debug):
    train_x, train_y, validation_x, validation_y, test_x, test_y = get_mnist_data(data_path, validation_ratio)
    height, width = train_x.shape[1:3]

    hidden_sizes = [100]
    output_size = 10
    latent_size = 32
    filter_channels=[32]

    tf.reset_default_graph()

    with tf.name_scope("placeholders"):
        x = tf.placeholder(tf.float32, (None, height, width, 1))
        y = tf.placeholder(tf.int64, (None,))

    # model = FullyConnectedClassifier(input_size=(height * width), output_size=10, hidden_sizes=hidden_output_sizes)
    model = CNNClassifier(height=28, width=28, input_channels=1, latent_size=latent_size, filter_channels=filter_channels, 
                            output_size=output_size, hidden_sizes=hidden_sizes)

    logits = model(x)
    predictions = tf.argmax(logits, 1)

    model_weights, _ = model.parameters

    loss = tf.losses.sparse_softmax_cross_entropy(y, logits)
    for weight in model_weights:
        loss += l2_coef * tf.nn.l2_loss(weight)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, predictions), tf.float32))
    opt = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    prev_best = None
    with tf.Session() as sess:
        if not debug:
            start_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            train_writer = tf.summary.FileWriter(logdir="./runs/{}_train".format(start_time))
            validation_writer = tf.summary.FileWriter(logdir="./runs/{}_validation".format(start_time))

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        for epoch in range(epochs):
            if not debug:
                _, train_summary, train_loss, train_accuracy = sess.run([opt, merged, loss, accuracy], 
                                                                        feed_dict={x: train_x, y: train_y})
                train_writer.add_summary(train_summary, epoch)
                
                validation_summary, validation_loss, validation_accuracy = sess.run([merged, loss, accuracy], 
                                                                                    feed_dict={x: validation_x, y: validation_y})
                validation_writer.add_summary(validation_summary, epoch)

                if not prev_best or validation_loss < prev_best:
                    save_path = saver.save(sess, model_path)
            else:
                _, train_loss, train_accuracy = sess.run([opt, loss, accuracy], 
                                                                    feed_dict={x: train_x, y: train_y})
                validation_loss, validation_accuracy = sess.run([loss, accuracy], 
                                                                    feed_dict={x: validation_x, y: validation_y})

            if (epoch + 1) % 1 == 0:
                print("Epoch: {} ==================".format(epoch + 1))
                print("Training - Loss: {}, Accuracy: {}".format(train_loss, train_accuracy))
                print("Validation - Loss: {}, Accuracy: {}".format(validation_loss, validation_accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/Users/bryan/Documents/sandbox/data/mnist.npz", 
                        help='The location of the data file')
    parser.add_argument('--validation_ratio', type=float, default=0.2,
                        help='Train validation split ratio')
    parser.add_argument('--lr', type=float, default=3e-4, 
                        help='The learning rate')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--dropout', type=float, default=0, 
                        help='The dropout rate')
    parser.add_argument('--l2_coef', type=float, default=0.1,
                        help='The coefficient for L2 regularization')
    parser.add_argument('--model_path', type=str, default="/Users/bryan/Documents/sandbox/saved_models/default.ckpt")
    parser.add_argument('--debug', action='store_true', 
                        help='Debug mode')
    args = parser.parse_args()

    train(args.data_path, args.validation_ratio, args.lr, args.epochs, args.dropout, args.l2_coef, 
            args.model_path, args.debug)