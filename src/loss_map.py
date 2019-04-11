from PIL import Image
import numpy as np
from array_display import display_array
from scipy.ndimage.morphology import binary_erosion
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_cdt
import tensorflow as tf
import time
from sklearn.preprocessing import normalize

def dist_from_edge(img):
    I = binary_erosion(img) # Interior mask
    C = img - I             # Contour mask
    out = C.astype(int)     # Setup o/p and assign cityblock distances
    out[I] = cdist(np.argwhere(C), np.argwhere(I), 'euclidean').min(0) + 1
    return out

def compute_loss_map(array):
    if len(array.shape) == 4:
        array = array[0,:,:,0]
    interior = binary_erosion(array)
    invert = np.logical_not(array).astype(int)
    #mult_mask = np.add(np.multiply(-1,array),invert)
    contour = array - interior
    dist = distance_transform_cdt(np.logical_not(contour).astype(int),metric="taxicab")
    #return np.multiply(dist, mult_mask)
    return dist

def dist_map_loss(y_true,y_pred):
    mult = tf.multiply(tf.abs(y_true),y_pred)
    print(tf.reduce_sum(mult).eval())
    print(tf.cast(tf.count_nonzero(mult), dtype=tf.float32).eval())
    return tf.divide(tf.reduce_sum(mult), tf.cast(tf.count_nonzero(mult).eval(), dtype=tf.float32))

def main():
    im = Image.open("../CharacterImages/4e00.png")
    im = im.resize((100, 100))
    arr = np.logical_not(np.asarray(im)).astype(int)
    loss_map = compute_loss_map(arr)
    display_array(loss_map)
    display_array(np.where(loss_map == 0, 1, 0))
    
def main2():
    save_loss_maps_as_images()

if __name__ == "__main__":
    start = time.time()
    main2()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")