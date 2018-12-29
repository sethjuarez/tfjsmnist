import os
import sys
import time
import math
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# pylint: disable-msg=E0611
from tensorflow.python.tools import freeze_graph as freeze
# pylint: enable-msg=E0611

############################################################
# Helpers                                                  #
############################################################
def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def check_dir(path, check=False):
    if check:
        assert os.path.exists(path), '{} does not exist!'.format(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        return Path(path).resolve()

def save_model(sess, export_path, output_node):
    # saving model
    checkpoint = str(export_path.joinpath('model.ckpt').resolve())

    saver = tf.train.Saver()
    saver.save(sess, checkpoint)

    # graph
    tf.train.write_graph(sess.graph_def, str(export_path), "model.pb", as_text=False)

    # freeze
    g = os.path.join(export_path, "model.pb")
    frozen = os.path.join(export_path, "digits.pb")

    freeze.freeze_graph(
        input_graph = g, 
        input_saver = "", 
        input_binary = True, 
        input_checkpoint = checkpoint, 
        output_node_names = output_node,
        restore_op_name = "",
        filename_tensor_name = "",
        output_graph = frozen,
        clear_devices = True,
        initializer_nodes = "")

    print("Model saved to " + frozen)

############################################################
# Digits Data                                              #
############################################################
class Digits:
    def __init__(self, data_dir, batch_size):
        # load MNIST data (if not available)
        self._data = os.path.join(data_dir, 'mnist.npz')
        self._train, self._test = tf.keras.datasets.mnist.load_data(path=self._data)
        self._batch_size = batch_size
        self._train_count = self._train[0].shape[0]
        self._size = self._train[0].shape[1] * self._train[0].shape[2]
        self._total = math.ceil((1. * self._train_count) / self._batch_size)

        self._testX = self._test[0].reshape(self._test[0].shape[0], self._size) / 255.
        self._testY = np.eye(10)[self._test[1]]

        self._trainX = self._train[0].reshape(self._train_count, self._size) / 255.
        self._trainY = self._train[1]

    def __iter__(self):
        # shuffle arrays
        p = np.random.permutation(self._trainX.shape[0])
        self._trainX = self._trainX[p]
        self._trainY = self._trainY[p]

        # reset counter
        self._current = 0

        return self

    def __next__(self):
        if self._current > self._train_count:
            raise StopIteration

        x = self._trainX[self._current : self._current + self._batch_size,:]
        y = np.eye(10)[self._trainY[self._current : self._current + self._batch_size]]

        if x.shape[0] == 0:
            raise StopIteration

        self._current += self._batch_size
        
        return x, y

    def __getitem__(self, index):
        index = 0 if index < 0 else index
        index = self._train_count - 1 if index > self._train_count else index
        x = self._trainX[index, :]
        y = self._trainY[index]

        return x, y

    @property
    def test(self):
        return self._testX, self._testY

    @property
    def total(self):
        return self._total

############################################################
# Model Definition                                         #
############################################################
def cnn_model(x):
    conv1 = tf.layers.conv2d(inputs=tf.reshape(x, [-1, 28, 28, 1]), 
                             filters=32, 
                             kernel_size=[5, 5], 
                             padding="same", 
                             activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    
    with tf.name_scope('Model'):
        pred = tf.layers.dense(inputs=dense, units=10, activation=tf.nn.softmax)
        return tf.identity(pred, name="prediction")

############################################################
# Training Loop                                            #
############################################################
def train_model(x, y, cost, optimizer, accuracy, learning_rate, batch_size, epochs, data_dir, outputs_dir, logs_dir):

    info('Initializing Devices')
    print(' ')
    
    # load MNIST data (if not available)
    digits = Digits(data_dir, batch_size)
    test_x, test_y = digits.test
    
    # Create a summary to monitor cost tensor
    tf.summary.scalar("cost", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(str(logs_dir), graph=tf.get_default_graph())

    # Initializing the variables
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        acc = 0.
        info('Training')

        # epochs to run
        for epoch in range(epochs):
            print("Epoch {}".format(epoch+1))
            avg_cost = 0.
            # loop over all batches
            for i, (train_x, train_y) in enumerate(digits):

                # Run optimization, cost, and summary
                _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                         feed_dict={x: train_x, y: train_y})

                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * digits.total + i)
                # Compute average loss
                avg_cost += c / digits.total
                print("\r    Batch {}/{} - Cost {:5.4f}".format(i+1, digits.total, avg_cost), end="")

            acc = accuracy.eval({x: test_x, y: test_y})
            print("\r    Cost: {:5.4f}, Accuracy: {:5.4f}\n".format(avg_cost, acc))
        
        # save model
        info("Saving Model")
        save_model(sess, outputs_dir, 'Model/prediction')

def main(settings):
    # resetting graph
    tf.reset_default_graph()

    # mnist data image of shape 28*28=784
    x = tf.placeholder(tf.float32, [None, 784], name='x')

    # 0-9 digits recognition => 10 classes
    y = tf.placeholder(tf.float32, [None, 10], name='y')

    # model
    hx = cnn_model(x)

    # accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hx, 1), tf.argmax(y, 1)), tf.float32))

    # cost / loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hx))

    # optimizer
    optimizer = tf.train.AdamOptimizer(settings.lr).minimize(cost)

    # training session
    train_model(x, y, cost, optimizer, accuracy, 
        settings.lr, settings.batch, settings.epochs, 
        settings.data, settings.outputs, settings.logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=100, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=0.001, type=float)
    parser.add_argument('-g', '--logs', help='log directory', default='logs')
    parser.add_argument('-o', '--outputs', help='output directory', default='outputs')
    args = parser.parse_args()

    args.data = check_dir(args.data).resolve()
    args.outputs = check_dir(args.outputs).resolve()
    args.logs = check_dir(args.logs).resolve()
    
    main(args)