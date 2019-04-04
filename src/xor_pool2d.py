import tensorflow as tf
from array_display import display_array
from PIL import Image
import numpy as np
from loss_map import compute_loss_map, dist_map_loss

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
  x = tf.where(sub > 0.5, tf.ones(shape), tf.zeros(shape))
  x = tf.cast(x, dtype=tf.int64)
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
  print(y.get_shape().as_list())

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
  y = tf.where(sub > 0.5, tf.ones(shape), tf.zeros(shape))
  y = tf.multiply(y,x)
  return y

def main():
  sess = tf.Session()
  with sess.as_default():
    im = Image.open("../CharacterImages/Regular/4e00.png")
    im = im.resize((100, 100))
    arr = np.reshape(np.asarray(im),(1,100,100,1))

    loss_map = compute_loss_map(np.logical_not(arr).astype(int))
    loss_map = np.reshape(loss_map, (1,100,100,1))
    y_true = tf.constant(loss_map, dtype=tf.float32)
    display_array(loss_map)

    y_pred = tf.constant(arr, dtype=tf.float32)
    y_pred = xor_pool2d3x3(y_pred)
    print(y_pred.get_shape().as_list())
    display_array(y_pred.eval())

    print(y_true)
    print(y_pred)
    loss = dist_map_loss(y_true, y_pred)
    print(loss.eval())

if __name__ == '__main__':
  main()