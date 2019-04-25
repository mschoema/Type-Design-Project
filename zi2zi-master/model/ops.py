# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf


def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon,
                                        scale=True, is_training=is_training, scope=scope)


def conv2d(x, output_filters, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="conv2d"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.get_variable('W', [kh, kw, shape[-1], output_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        Wconv = tf.nn.conv2d(x, W, strides=[1, sh, sw, 1], padding='SAME')

        biases = tf.get_variable('b', [output_filters], initializer=tf.constant_initializer(0.0))
        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b


def deconv2d(x, output_shape, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="deconv2d"):
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        input_shape = x.get_shape().as_list()
        W = tf.get_variable('W', [kh, kw, output_shape[-1], input_shape[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape,
                                        strides=[1, sh, sw, 1])

        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv_plus_b = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv_plus_b


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def fc(x, output_size, stddev=0.02, scope="fc"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.get_variable("W", [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, W) + b


def init_embedding(size, dimension, stddev=0.01, scope="embedding"):
    with tf.variable_scope(scope):
        return tf.get_variable("E", [size, 1, 1, dimension], tf.float32,
                               tf.random_normal_initializer(stddev=stddev))


def conditional_instance_norm(x, ids, labels_num, mixed=False, scope="conditional_instance_norm"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        batch_size, output_filters = shape[0], shape[-1]
        scale = tf.get_variable("scale", [labels_num, output_filters], tf.float32, tf.constant_initializer(1.0))
        shift = tf.get_variable("shift", [labels_num, output_filters], tf.float32, tf.constant_initializer(0.0))

        mu, sigma = tf.nn.moments(x, [1, 2], keep_dims=True)
        norm = (x - mu) / tf.sqrt(sigma + 1e-5)

        batch_scale = tf.reshape(tf.nn.embedding_lookup([scale], ids=ids), [batch_size, 1, 1, output_filters])
        batch_shift = tf.reshape(tf.nn.embedding_lookup([shift], ids=ids), [batch_size, 1, 1, output_filters])

        z = norm * batch_scale + batch_shift
        return z

def xor_pool2d2x2(x, padding='SAME', name='pool'):
  batch_size, height, width, num_channels = x.get_shape().as_list()
  shape = (batch_size, height, width, num_channels)
  pad_bottom = height%2
  pad_right = width%2
  height_div2 = height + pad_bottom
  width_div2 = width + pad_right
  if(padding=='SAME'):
    x = tf.pad(x, [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]], "CONSTANT")

  _, height, width, _ = x.get_shape().as_list()
  offsets_y = tf.range(0, height, 1)
  offsets_x = tf.range(0, width, 1)

  x = tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT", constant_values=1)

  sub_y0 = tf.gather(x, offsets_y, axis=1)
  sub_y1 = tf.gather(x, offsets_y + 1, axis=1)

  sub_00 = tf.gather(sub_y0, offsets_x, axis=2)
  sub_00 = tf.reshape(sub_00, shape)
  sub_01 = tf.gather(sub_y0, offsets_x + 1, axis=2)
  sub_01 = tf.reshape(sub_01, shape)

  sub_10 = tf.gather(sub_y1, offsets_x, axis=2)
  sub_10 = tf.reshape(sub_10, shape)
  sub_11 = tf.gather(sub_y1, offsets_x + 1, axis=2)
  sub_11 = tf.reshape(sub_11, shape)

  maxi = tf.maximum(tf.maximum(sub_00,sub_01),tf.maximum(sub_10,sub_11))
  mini = tf.minimum(tf.minimum(sub_00,sub_01),tf.minimum(sub_10,sub_11))

  sub = tf.subtract(maxi, mini)
  x = tf.where(sub > 0.2, tf.ones(shape), tf.zeros(shape))
  return x

def xor_pool2d3x3(x, padding='SAME', name='pool'):
  batch_size, height, width, num_channels = x.get_shape().as_list()
  shape = (batch_size, height, width, num_channels)
  pad_bottom = height%2
  pad_right = width%2
  height_div2 = height + pad_bottom
  width_div2 = width + pad_right
  if(padding=='SAME'):
    y = tf.pad(x, [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]], "CONSTANT")

  _, height, width, _ = y.get_shape().as_list()
  offsets_y = tf.range(1, height+1, 1)
  offsets_x = tf.range(1, width+1, 1)

  y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT", constant_values=1)

  sub_y0 = tf.gather(y, offsets_y, axis=1)
  sub_y1 = tf.gather(y, offsets_y + 1, axis=1)
  sub_y2 = tf.gather(y, offsets_y - 1, axis=1)

  sub_00 = tf.gather(sub_y0, offsets_x, axis=2)
  sub_00 = tf.reshape(sub_00, shape)
  sub_01 = tf.gather(sub_y0, offsets_x + 1, axis=2)
  sub_01 = tf.reshape(sub_01, shape)
  sub_02 = tf.gather(sub_y0, offsets_x - 1, axis=2)
  sub_02 = tf.reshape(sub_02, shape)

  sub_10 = tf.gather(sub_y1, offsets_x, axis=2)
  sub_10 = tf.reshape(sub_10, shape)
  sub_11 = tf.gather(sub_y1, offsets_x + 1, axis=2)
  sub_11 = tf.reshape(sub_11, shape)
  sub_12 = tf.gather(sub_y1, offsets_x - 1, axis=2)
  sub_12 = tf.reshape(sub_12, shape)

  sub_20 = tf.gather(sub_y2, offsets_x, axis=2)
  sub_20 = tf.reshape(sub_20, shape)
  sub_21 = tf.gather(sub_y2, offsets_x + 1, axis=2)
  sub_21 = tf.reshape(sub_21, shape)
  sub_22 = tf.gather(sub_y2, offsets_x - 1, axis=2)
  sub_22 = tf.reshape(sub_22, shape)

  maxi = tf.maximum(tf.maximum(tf.maximum(tf.maximum(sub_00,sub_01),tf.maximum(sub_10,sub_11)),tf.maximum(tf.maximum(sub_20,sub_21),tf.maximum(sub_12,sub_22))),sub_02)
  mini = tf.minimum(tf.minimum(tf.minimum(tf.minimum(sub_00,sub_01),tf.minimum(sub_10,sub_11)),tf.minimum(tf.minimum(sub_20,sub_21),tf.minimum(sub_12,sub_22))),sub_02)

  sub = tf.subtract(maxi, mini)
  y = tf.where(sub > 0.2, tf.ones(shape), tf.zeros(shape))
  y = tf.multiply(y,x)
  return y

@tf.custom_gradient
def dist_map_loss(y_true,y_pred):
  mult = tf.multiply(tf.abs(y_true),y_pred)
  loss = tf.reduce_mean(mult)
  def grad(dy):
    return tf.zeros_like(y_true), y_true
  return loss, grad