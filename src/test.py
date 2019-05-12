import time
import os
import numpy as np
from PIL import Image
from array_display import display_array
import pickle
from io import BytesIO
import scipy.misc as misc
import tensorflow as tf
from ops import *


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    mat = misc.imread(img).astype(np.float)
    side = int(mat.shape[1] / 3)
    assert side * 3 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:side*2]  # source
    img_C = mat[:, side*2:]  # loss map

    return img_A, img_B, img_C

def load_pickled_examples(obj_path):
        with open(obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 1000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples

def process(img):
    img = bytes_to_file(img)
    try:
        img_A, img_B, loss_map = read_split_image(img)
        img_A = normalize_image(img_A)
        img_B = normalize_image(img_B)
        loss_map = loss_map*img_A
        return np.concatenate([img_A[:,:,np.newaxis], img_B[:,:,np.newaxis], loss_map[:,:,np.newaxis]], axis=2)
    finally:
        img.close()

def np_xor_pooling(x):
    x = np.round(x)
    shape = x.shape
    batch_size, height, width, num_channels = shape
    pad_bottom = height%2
    pad_right = width%2
    height_div2 = height + pad_bottom
    width_div2 = width + pad_right
    y = np.pad(x, ((0, 0), (0, pad_bottom), (0, pad_right), (0, 0)), "constant")

    _, height, width, _ = y.shape
    offsets_y = np.arange(1, height+1, 1)
    offsets_x = np.arange(1, width+1, 1)

    y = np.pad(y, ((0, 0), (1, 1), (1, 1), (0, 0)), "constant")

    sub_y0 = np.take(y, offsets_y, axis=1)
    sub_y1 = np.take(y, offsets_y + 1, axis=1)
    sub_y2 = np.take(y, offsets_y - 1, axis=1)

    sub_00 = np.take(sub_y0, offsets_x, axis=2)
    sub_00 = np.reshape(sub_00, shape)
    sub_01 = np.take(sub_y0, offsets_x + 1, axis=2)
    sub_01 = np.reshape(sub_01, shape)
    sub_02 = np.take(sub_y0, offsets_x - 1, axis=2)
    sub_02 = np.reshape(sub_02, shape)

    sub_10 = np.take(sub_y1, offsets_x, axis=2)
    sub_10 = np.reshape(sub_10, shape)
    sub_11 = np.take(sub_y1, offsets_x + 1, axis=2)
    sub_11 = np.reshape(sub_11, shape)
    sub_12 = np.take(sub_y1, offsets_x - 1, axis=2)
    sub_12 = np.reshape(sub_12, shape)

    sub_20 = np.take(sub_y2, offsets_x, axis=2)
    sub_20 = np.reshape(sub_20, shape)
    sub_21 = np.take(sub_y2, offsets_x + 1, axis=2)
    sub_21 = np.reshape(sub_21, shape)
    sub_22 = np.take(sub_y2, offsets_x - 1, axis=2)
    sub_22 = np.reshape(sub_22, shape)

    maxi = np.maximum(np.maximum(np.maximum(np.maximum(sub_00,sub_01),np.maximum(sub_10,sub_11)),np.maximum(np.maximum(sub_20,sub_21),np.maximum(sub_12,sub_22))),sub_02)
    mini = np.minimum(np.minimum(np.minimum(np.minimum(sub_00,sub_01),np.minimum(sub_10,sub_11)),np.minimum(np.minimum(sub_20,sub_21),np.minimum(sub_12,sub_22))),sub_02)

    sub = np.subtract(maxi, mini)
    out = np.where(sub == 1, np.ones_like(sub), np.zeros_like(sub))
    out = np.multiply(out,x)

    def grad(dy):

        y_00 = np.roll(out, shift=[-1,-1], axis=[1,2])
        y_01 = np.roll(out, shift=[-1,0], axis=[1,2])
        y_02 = np.roll(out, shift=[-1,1], axis=[1,2])
        y_0 = np.add(np.add(y_00, y_01), y_02)
        y_10 = np.roll(out, shift=[0,-1], axis=[1,2])
        y_11 = out
        y_12 = np.roll(out, shift=[0,1], axis=[1,2])
        y_1 = np.add(np.add(y_10, y_11), y_12)
        y_20 = np.roll(out, shift=[1,-1], axis=[1,2])
        y_21 = np.roll(out, shift=[1,0], axis=[1,2])
        y_22 = np.roll(out, shift=[1,1], axis=[1,2])
        y_2 = np.add(np.add(y_20, y_21), y_22)

        y_out = np.add(np.add(y_0, y_1), y_2)
        # return np.multiply(dy, out)
        return dy

    return out, grad

def np_dist_map_loss(y_true,y_pred):
  mult = np.multiply(np.abs(y_true),y_pred)
  loss = np.mean(mult)
  def grad():
    return y_true
  return loss, grad


def sigmoid_array(x):

    sigm = 1. / (1. + np.exp(-x))

    def grad(dy):
        return np.multiply(dy,np.multiply(sigm,(1. - sigm)))

    return sigm, grad

def main():
    examples = load_pickled_examples("../inputFiles/experiment/train.obj")
    img = examples[0]
    arrays = process(img[1])
    arr0 = arrays[:,:,0]
    arr1 = arrays[:,:,1]
    arr2 = arrays[:,:,2]
    # display_array(arr0)
    # display_array(arr1)
    # display_array(arr2)
    arr = -arr1[np.newaxis,:,:,np.newaxis]
    arr2 = np.clip(arr2[np.newaxis,:,:,np.newaxis],-1, 1)
    gamma = 0.5
    count = 0
    with tf.Session() as sess:
        while(True):
            out_arr, sigm_grad_fct = sigmoid_array(arr)
            edge_arr, xor_grad_fct = np_xor_pooling(out_arr)
            loss, loss_grad_fct = np_dist_map_loss(arr2,edge_arr)
            loss_grad = loss_grad_fct()
            edge_grad = xor_grad_fct(loss_grad)
            out_grad = sigm_grad_fct(edge_grad)
            arr = arr - gamma * out_grad

            if count%50 == 0:
                print(loss)
                display_array(np.round(out_arr))
                display_array(out_grad)
                display_array(arr)
            count += 1


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")