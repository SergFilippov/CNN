#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy

NUM_CLASSES = 10
IMAGE_SIZE = 28


class Net:
    def __init__(self):
        pass

    def inference(self, images, train=False):
        """ Описывает структуру сети
        Args:
            images: placeholder, вход сети
        Returns:
            fcl2_hidden: Tensor, выход сети
        """
        # Первый сверточный слой
        conv1_conv2d = tf.nn.conv2d(images, self.__conv1_weights, strides=[1, 1, 1, 1],
                                    padding='SAME') + self.__conv1_biases
        conv1_relu = tf.nn.relu(conv1_conv2d)
        conv1_pool = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME')

        # Второй сверточный слой
        conv2_conv2d = tf.nn.conv2d(conv1_pool, self.__conv2_weights, strides=[1, 1, 1, 1],
                                    padding='SAME') + self.__conv2_biases
        conv2_relu = tf.nn.relu(conv2_conv2d)
        conv2_pool = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME')
        conv2_shape = conv2_pool.get_shape().as_list()
        conv2_pool = tf.reshape(conv2_pool,
                                [conv2_shape[0], conv2_shape[1] * conv2_shape[2] * conv2_shape[3]])

        # Первый полносвязный слой
        fcl1_hidden = tf.matmul(conv2_pool, self.__fcl1_weights) + self.__fcl1_biases
        fcl1_relu = tf.nn.relu(fcl1_hidden)

        if train:
            fcl1_relu = tf.nn.dropout(fcl1_relu, 0.5)

        # Второй полносвязный слой
        fcl2_hidden = tf.matmul(fcl1_relu, self.__fcl2_weights) + self.__fcl2_biases
        return fcl2_hidden

    def train(self, train_data, train_labels, num_epochs=10, batch_size=64):
        """ Обучение нейронной сети
        Args:
            train_data: numpy массив {size, x, y, channels}
            train_labels: numpy массив {size}
        """

        train_size = train_labels.shape[0]

        train_data_node = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, 1))
        train_labels_node = tf.placeholder(tf.int64, shape=(batch_size,))

        outputs = self.inference(train_data_node)
        train_prediction = tf.nn.softmax(outputs)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, train_labels_node))
        regularizers = (tf.nn.l2_loss(self.__fcl1_weights) + tf.nn.l2_loss(self.__fcl1_biases) +
                        tf.nn.l2_loss(self.__fcl2_weights) + tf.nn.l2_loss(self.__fcl2_biases))
        loss += 5e-4 * regularizers

        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.01, batch * batch_size, train_size, 0.95, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

        # Создание сессии
        with tf.Session() as sess:
            # Инициализация всех переменных
            tf.initialize_all_variables().run()
            print('Initialized!')
            # Цикл обучения
            for step in xrange(int(num_epochs * train_size) // batch_size):
                # Вычисление смещения текущей выборки в файле данных
                offset = (step * batch_size) % (train_size - batch_size)
                batch_data = train_data[offset:(offset + batch_size), ...]
                batch_labels = train_labels[offset:(offset + batch_size)]

                # Запуск вычислительного графа
                _, predictions = sess.run([optimizer, train_prediction],
                                          feed_dict={train_data_node: batch_data, train_labels_node: batch_labels})
                if step % 100 == 0:
                    print('Step %d (epoch %.2f)' % (step, float(step) * batch_size / train_size))
                    error = 100.0 - (
                        100.0 * numpy.sum(numpy.argmax(predictions, 1) == batch_labels) / predictions.shape[0])
                    print('Test error: %.3f%%' % error)

            # Сохранение обученной сети в файл
            if not tf.gfile.Exists('save'):
                tf.gfile.MakeDirs('save')
            saver = tf.train.Saver()
            saver.save(sess, "save/model.ckpt")
            tf.train.write_graph(sess.graph_def, "save/graph", "graph.pb", as_text=False)
        return

    # Коэффициенты нейронной сети:
    # Первый сверточный слой
    __conv1_weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    __conv1_biases = tf.Variable(tf.zeros([32]))

    # Второй сверточный слой
    __conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    __conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    # Первый полносвязный слой
    __fcl1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                                                     stddev=0.1))
    __fcl1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    # Второй полносвязный слой
    __fcl2_weights = tf.Variable(tf.truncated_normal([512, NUM_CLASSES],
                                                     stddev=0.1))
    __fcl2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))

