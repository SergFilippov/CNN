#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy
from PIL import Image

IMAGE_SIZE = 28


# Представление данных в виде тензора [image index, y, x, channels]
def open_data(filename, num_images):
    if not tf.gfile.Exists(filename):
        print("Can not read data " + filename)
        return
    print('Opening', filename)
    with open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (255 / 2.0)) / 255   # Представление цвета в диапазоне [-0.5; 0.5]
        data.resize(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


# Представление меток в виде вектора int64
def open_labels(filename, num_images):
    if not tf.gfile.Exists(filename):
        print("Can not read labels " + filename)
        return
    print('Opening', filename)
    with open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
    return labels


# Перевод изображения в формат Mnist
def read_image(filename):
    if not tf.gfile.Exists(filename):
        print("Can not read image " + filename)
        return
    im = Image.open(filename).convert('L')  # Формат файла может быть любым
    im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    pix = im.load()

    buf = []
    for x in range(IMAGE_SIZE):
        for y in range(IMAGE_SIZE):
            buf.append(float(pix[y, x]))
    data = numpy.array(buf, dtype=numpy.uint8).astype(numpy.float32)
    data.resize((1, IMAGE_SIZE, IMAGE_SIZE, 1))
    data = -(data - (255 / 2.0)) / 255
    return data
