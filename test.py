#!/usr/bin/env python
# -*- coding:utf-8 -*-

import conv_net
import read_data
import numpy
import tensorflow as tf

TEST_SIZE = 10000
NUM_LABELS = 10
IMAGE_SIZE = 28


def error_rate(predictions, labels):
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])


if __name__ == '__main__':
    test_data = read_data.open_data('data/t10k-images.idx3-ubyte', TEST_SIZE)
    test_labels = read_data.open_labels('data/t10k-labels.idx1-ubyte', TEST_SIZE)

    eval_data = tf.placeholder(tf.float32, shape=(TEST_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
    net = conv_net.Net()
    eval_prediction = tf.nn.softmax(net.inference(eval_data))



    with tf.Session() as sess:
        # Загрузка всех параметров из файла
        saver = tf.train.Saver()
        saver.restore(sess, 'save/model.ckpt')
        print('Initialized!')

        # Вывод результата
        test_error = error_rate(sess.run(eval_prediction, feed_dict={eval_data: test_data}), test_labels)

        print('Test error: %.1f%%' % test_error)

    print('Complited!')