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

def main():
    tf.enable_eager_execution()
    examples = load_pickled_examples("../inputFiles/experiment/train.obj")
    img = examples[0]
    arrays = process(img[1])
    arr0 = arrays[:,:,0]
    arr1 = arrays[:,:,1]
    arr2 = arrays[:,:,2]
    display_array(arr2)
    arr = arr1[np.newaxis,:,:,np.newaxis]
    with tf.Session() as sess:
        #arr = np.random.randint(2,size=(1,256,256,1))
        out_tensor = tf.Variable(arr,dtype=tf.float32)
        sess.run(out_tensor.initializer)
        loss_tensor = tf.Variable(arr2,dtype=tf.float32)
        sess.run(loss_tensor.initializer)
        out_tensor = (-1.*out_tensor + 1.)/2.
        edge_tensor = xor_pool2d3x3(out_tensor)
        edge_arr = edge_tensor.eval()
        print(np.min(edge_arr))
        print(np.max(edge_arr))
        display_array(edge_arr)
        loss = dist_map_loss(loss_tensor,edge_tensor)
        print(loss.eval())
        grad_tensor = tf.gradients(loss, [out_tensor])[0]
        grad_arr = grad_tensor.eval()
        print(np.min(grad_arr))
        print(np.max(grad_arr))
        display_array(grad_arr)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")