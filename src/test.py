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
    image2 = image.transpose(Image.FLIP_LEFT_RIGHT)
    image3 = image.transpose(Image.FLIP_TOP_BOTTOM)
    image4 = image2.transpose(Image.FLIP_TOP_BOTTOM)
    image5 = image.transpose(Image.ROTATE_180)
    image.show()
    image2.show()
    image3.show()
    image4.show()
    image5.show()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")