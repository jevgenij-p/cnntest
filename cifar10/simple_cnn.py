import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layers import conv_layer, max_pool_2x2, full_layer

DATA_PATH = "../cifar-10-batches-py/"
LOG_PATH = "log"
BATCH_SIZE = 50
STEPS = 500000

def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    return dict

class CifarLoader(object):
    """
    Load and manage the CIFAR dataset.
    """
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)\
            .astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size], \
               self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def random_batch(self, batch_size):
        n = len(self.images)
        ix = np.random.choice(n, batch_size)
        return self.images[ix], self.labels[ix]

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)])\
            .load()
        self.test = CifarLoader(["test_batch"]).load()

def run_simple_net():
    cifar = CifarDataManager()

    def test(sess, step):
        X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
        Y = cifar.test.labels.reshape(10, 1000, 10)
        acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0})
                       for i in range(10)])
        print("Step {0}, Accuracy: {1:.4}%".format(step, acc * 100))

    with tf.name_scope("placeholders"):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope("convolution1"):
        conv1 = conv_layer(x, shape=[5, 5, 3, 32])
        conv1_pool = max_pool_2x2(conv1)

    with tf.name_scope("convolution2"):
        conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
        conv2_pool = max_pool_2x2(conv2)

    with tf.name_scope("convolution3"):
        conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 128])
        conv3_pool = max_pool_2x2(conv3)
        conv3_flat = tf.reshape(conv3_pool, [-1, 4 * 4 * 128])
        conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

    with tf.name_scope("fully_connected"):
        full_1 = tf.nn.relu(full_layer(conv3_drop, 512))
        tf.summary.histogram('activations', full_1)
        
        full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
        y_conv = full_layer(full1_drop, 10)

    with tf.name_scope("cross_entropy"):
        with tf.name_scope("total"):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,
                                                                                   labels=y_))
        tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_PATH, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            batch = cifar.train.next_batch(BATCH_SIZE)
            _, summary = sess.run([train_step, merged], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            if i % 500 == 0:
                test(sess, i)
                train_writer.add_summary(summary, i)

        test(sess, i)

def main():
    if tf.gfile.Exists(LOG_PATH):
        tf.gfile.DeleteRecursively(LOG_PATH)
        tf.gfile.MakeDirs(LOG_PATH)
    run_simple_net()

if __name__ == "__main__":
    main()
