from PIL import Image, ImageChops
import numpy as np
from layouts import BoundingBox
import itertools
import sys
from matplotlib import pyplot as plt

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

def insideImage(pos):
    (x,y) = pos
    return x >= 0 and x < 1000 and y >= 0 and y < 1000

def floodFill(array, x, y, label):
    queue = Queue()
    queue.enqueue((x,y))
    while not queue.isEmpty():
        (x,y) = queue.dequeue()
        if insideImage((x,y)) and array[x][y] == 1:
            array[x][y] = label
            queue.enqueue((x+1,y))
            queue.enqueue((x,y+1))
            queue.enqueue((x,y-1))
            queue.enqueue((x-1,y))

def regionLabeling(array):
    regions = []
    m = 2
    for i in range(1000):
        for j in range(1000):
            if (array[i][j] == 1):
                regions.append(m)
                floodFill(array,i,j,m)
                m += 1
    return regions

def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def findBoundingBox(array, label):
    region = np.where(array == label)
    ymin, ymax = np.min(region[0]), np.max(region[0])
    xmin, xmax = np.min(region[1]), np.max(region[1])
    return BoundingBox(xmin, ymin, xmax-xmin, ymax-ymin)

def findBoxes(image):
    arr = np.logical_not(np.asarray(image)).astype(int)
    regions = regionLabeling(arr)
    boxes = []
    for i in range(2, 2 + regions):
        box = findBoundingBox(arr,i)
        boxes.append(box)
    return boxes

def findBestBox(arr, regions, charImage):
    arr2 = np.logical_not(np.asarray(charImage)).astype(int)
    bb = findBoundingBox(arr2, 1)
    char = charImage.crop(bb.getBoxOutline())
    minError = 99999999
    bestBox = None
    for L in range(len(regions)):
        for subset in itertools.combinations(regions, L+1):
            arrSubset = np.invert(np.isin(arr, list(subset)))
            box = findBoundingBox(np.logical_not(arrSubset).astype(int), 1)
            if box.getSize()[0] == 0 or box.getSize()[1] == 0:
                continue
            subImage = img_frombytes(arrSubset)
            im1 = subImage.crop(box.getBoxOutline())
            im2 = char.resize(box.getSize())
            arr1 = np.asarray(im1)
            arr2 = np.asarray(im2)
            arrXor = np.logical_xor(arr1,arr2)
            (dx, dy) = box.getSize()
            error = np.count_nonzero(arrXor) / (dx * dy)
            if error < minError:
                minError = error
                bestBox = box
    return (bestBox, minError)

def areSame(chars):
    res = True
    for i in range(len(chars) - 1):
        res = res and chars[i] == chars[i+1]
    return res

def findBestBoxes(image, comps):
    arr = np.logical_not(np.asarray(image)).astype(int)
    regions = regionLabeling(arr)
    totError = 0
    if len(regions) == len(comps) and areSame(comps):
        boxes = []
        for i in range(2, 2 + len(regions)):
            box = findBoundingBox(arr,i)
            boxes.append(box)
    else:
        boxes = []
        for comp in comps:
            (box, error) = findBestBox(arr, regions, comp)
            boxes.append(box)
            totError += error
    return (boxes, totError)

def main(c, c1, c2):
    im = Image.open("../CharacterImages/" + c + ".png")
    im1 = Image.open("../CharacterImages/" + c1 + ".png")
    im2 = Image.open("../CharacterImages/" + c2 + ".png")
    arr1 = np.logical_not(np.asarray(im1)).astype(int)
    arr2 = np.logical_not(np.asarray(im2)).astype(int)
    bb1 = findBoundingBox(arr1,1)
    bb2 = findBoundingBox(arr2,1)
    char1 = im1.crop(bb1.getBoxOutline())
    char2 = im2.crop(bb2.getBoxOutline())
    char1.save("../testFiles/char1.png")
    char2.save("../testFiles/char2.png")
    boxes, err = findBestBoxes(im,[im1,im2])
    box1 = boxes[0]
    box2 = boxes[1]
    print(box1)
    print(box2)
    cut1 = im.crop(box1.getBoxOutline())
    cut2 = im.crop(box2.getBoxOutline())
    cut1.save("../testFiles/cut1.png")
    cut2.save("../testFiles/cut2.png")
    char1 = char1.resize(box1.getSize())
    char2 = char2.resize(box2.getSize())
    out = Image.new("1",(1000,1000),1)
    out.paste(char1, box1.getBoxOutline(), ImageChops.invert(char1))
    out.paste(char2, box2.getBoxOutline(), ImageChops.invert(char2))
    out.save("../testFiles/out.png")

if __name__ == "__main__":
    if (len(sys.argv) == 4 ):
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main("4ead", "4ea0", "53e3")