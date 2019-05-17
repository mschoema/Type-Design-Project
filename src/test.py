import time
import os
import numpy as np
from PIL import Image
from array_display import display_array
import pickle
from io import BytesIO
import scipy.misc as misc
import tensorflow as tf


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
        return np.concatenate([img_A[:,:,np.newaxis], img_B[:,:,np.newaxis]], axis=2)
    finally:
        img.close()

def main():
    examples = load_pickled_examples("../inputFiles/experiment/train.obj")
    img = examples[0]
    arrays = process(img[1])
    arr0 = arrays[:,:,0]
    arr1 = arrays[:,:,1]
    display_array(arr0)
    display_array(arr1)
    arr = np.multiply(arr0,arr1)
    display_array(arr)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")