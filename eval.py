#!/usr/bin/env python
# -*- coding:utf-8 -*-

import read_data
import conv_net
import tensorflow as tf
import numpy

IMAGE_SIZE = 28

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_file', 'test.png',
                           """Path to image file.""")


if __name__ == '__main__':
    eval_data = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, 1))
    net = conv_net.Net()
    eval_prediction = tf.nn.softmax(net.inference(eval_data))

    with tf.Session() as sess:
        # Загрузка всех параметров из файла
        saver = tf.train.Saver()
        saver.restore(sess, 'save/model.ckpt')
        print('Initialized!')

        if not tf.gfile.Exists(FLAGS.image_file):
            print("Can not read data " + FLAGS.image_file)
        else:
            prediction = sess.run(eval_prediction, feed_dict={eval_data: read_data.read_image(FLAGS.image_file)})
            print(numpy.argmax(prediction, 1))