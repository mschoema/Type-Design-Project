import time
import os
import glob
import numpy as np
from PIL import Image, ImageDraw
from array_display import display_array
import pickle
from io import BytesIO
import scipy.misc as misc
import tensorflow as tf
from  class_definitions import BoundingBox
from layouts import applyLayout, applyPreciseLayout
from findPosition_v3 import regionLabeling, img_frombytes
from findPosition_v2 import findBoxes
from xor_pool2d import xor_pool2d3x3
from loss_map import compute_loss_map

def main():
    image1 = Image.open("../characterImages/Regular/4ea0.png")
    image2 = Image.open("../characterImages/Regular/51e0.png")
    images = [image1, image2]
    box1 = BoundingBox(69, 30, 866, 266)
    box2 = BoundingBox(43, 404, 918, 557)
    boxes = [box1, box2]
    image3 = applyPreciseLayout(boxes, images)
    image3.save("../testFiles/Regular_4ea2.png")

def main8():
    image1 = Image.open("../characterImages/Regular/4ea0.png")
    image2 = Image.open("../characterImages/Regular/53e3.png")
    image3 = Image.open("../characterImages/Regular/5196.png")
    image4 = Image.open("../characterImages/Regular/51e0.png")
    image = Image.open("../characterImages/Regular/4eae.png")
    boxes = findBoxes(image)
    images = [image1, image2, image3, image4]
    image5 = applyPreciseLayout(boxes, images)
    image5.save("../testFiles/4eae_preciseDef.png")
    image6 = applyLayout(26, images)
    image6.save("../testFiles/4eae_roughDef.png")

def main7():
    image = Image.open("../characterImages/Regular/4eae.png")
    boxes = findBoxes(image)
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box.getBoxOutline(), outline='red')
    image.save("../testFiles/4eae_patches.png")


def main6():
    font_dir = "../inputFiles/font/"
    test = os.listdir(font_dir)
    print(test)


def main5():
    image = Image.open("../characterImages/Regular/4ea2.png")
    with tf.Session() as sess:
        arr = np.logical_not(np.asarray(image)).astype(np.float32)[np.newaxis,:,:,np.newaxis]
        tensor = tf.constant(arr)
        (batch_size, h, w, d) = tensor.shape
        edges = tf.image.sobel_edges(tensor)
        edges = tf.reshape(edges, (batch_size, h, w, 2*d))
        edges_arr = edges.eval()
        sobel_x = img_frombytes(np.logical_not(edges_arr[0,:,:,0]).astype(int))
        sobel_y = img_frombytes(np.logical_not(edges_arr[0,:,:,1]).astype(int))
        sobel_x.save("../testFiles/4ea2_sobel_x.png")
        sobel_y.save("../testFiles/4ea2_sobel_y.png")

def main4():
    image = Image.open("../characterImages/Regular/4ea2.png")
    arr = np.logical_not(np.asarray(image)).astype(int)
    loss_map = compute_loss_map(arr)
    arr = compute_loss_map(arr).astype(float)
    arr = arr / np.amax(arr)
    arr = (arr * 255).astype(np.uint8)
    image2 = Image.fromarray(arr, mode='L')
    image2.save("../testFiles/4ea2_distance_map.png")

def main3():
    image = Image.open("../characterImages/Regular/4ea2.png")
    with tf.Session() as sess:
        arr = np.logical_not(np.asarray(image)).astype(np.float32)[np.newaxis,:,:,np.newaxis]
        tensor = tf.constant(arr)
        edges = xor_pool2d3x3(tensor)
        edges_arr = edges.eval()
        image2 = img_frombytes(np.logical_not(edges_arr[0,:,:,0]).astype(int))
        image2.save("../testFiles/4ea2_edges.png")

def main2():
    image1 = Image.open("../characterImages/Regular/4ea2.png")
    image2 = Image.open("../characterImages/Regular/51e0.png")
    images = [image1, image2]
    box1 = BoundingBox(69, 30, 866, 266)
    box2 = BoundingBox(43, 404, 918, 557)
    boxes = [box1, box2]
    image3 = applyPreciseLayout(boxes, images)
    image3.save("../testFiles/Regular_4ea2_preciseDef.png")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")