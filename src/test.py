import time
import os
import numpy as np
from PIL import Image
from array_display import display_array
import pickle
from io import BytesIO
import scipy.misc as misc
import tensorflow as tf

def main():
    image = Image.open("../outputFiles/Regular/rough_1000/4e01.png")
    arr = np.asarray(image)
    arr2 = np.fliplr(arr)
    arr3 = np.flipud(arr)
    arr4 = np.flipud(arr2)
    arr5 = np.rot90(arr, k=2)
    display_array(arr)
    display_array(arr2)
    display_array(arr3)
    display_array(arr4)
    display_array(arr5)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")