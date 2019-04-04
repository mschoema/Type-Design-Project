from PIL import Image, ImageChops
import numpy as np
from layouts import BoundingBox
import itertools
import sys

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
    m = 2
    for i in range(1000):
        for j in range(1000):
            if (array[i][j] == 1):
                floodFill(array,i,j,m)
                m += 1
    return m-2

def areSame(chars):
    res = True
    for i in range(len(chars) - 1):
        res = res and chars[i] == chars[i+1]
    return res

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

def combineBoxes(boxes):
    Xmin, Xmax = 999, 0
    Ymin, Ymax = 999, 0
    for box in boxes:
        (xmin, ymin, xmax, ymax) = box.getBoxOutline()
        if xmin < Xmin:
            Xmin = xmin
        if xmax > Xmax:
            Xmax =xmax
        if ymin < Ymin:
            Ymin = ymin
        if ymax > Ymax:
            Ymax = ymax
    return BoundingBox(Xmin, Ymin, Xmax-Xmin, Ymax-Ymin)

def findBestBox(image, boxes, charImage):
    arr = np.logical_not(np.asarray(charImage)).astype(int)
    bb = findBoundingBox(arr, 1)
    char = charImage.crop(bb.getBoxOutline())
    minError = 99999999
    bestBox = None
    for L in range(len(boxes)):
        for subset in itertools.combinations(boxes, L+1):
            box = combineBoxes(list(subset))
            if box.getSize()[0] == 0 or box.getSize()[1] == 0:
                continue
            im1 = char.resize(box.getSize())
            im2 = image.crop(box.getBoxOutline())
            arr1 = np.asarray(im1)
            arr2 = np.asarray(im2)
            arrXor = np.logical_xor(arr1,arr2)
            (dx, dy) = box.getSize()
            error = np.count_nonzero(arrXor) / (dx * dy)
            if error < minError:
                minError = error
                bestBox = box
    return (bestBox, minError)

def computeError(image, box, charImage):
    arr = np.logical_not(np.asarray(charImage)).astype(int)
    bb = findBoundingBox(arr, 1)
    char = charImage.crop(bb.getBoxOutline())
    im1 = char.resize(box.getSize())
    im2 = image.crop(box.getBoxOutline())
    arr1 = np.asarray(im1)
    arr2 = np.asarray(im2)
    arrXor = np.logical_xor(arr1,arr2)
    (dx, dy) = box.getSize()
    return np.count_nonzero(arrXor) / (dx * dy)


def matrixElim(image, baseBoxes, compImgs):
    n = len(baseBoxes)
    errArr = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            errArr[i,j] = computeError(image,baseBoxes[i],compImgs[j])
    print(errArr)
    mean = errArr.mean(axis=1)
    errArr = errArr - mean[:,None]
    print(errArr)
    boxes = [None]*n
    boxTaken = [False]*n
    for i in range(n):
        ind = np.unravel_index(np.argmin(errArr, axis=None), errArr.shape)
        count = ind[1]
        boxPos = 0
        print(count)
        while count > 0:
            if not boxTaken[boxPos]:
                count -= 1
            boxPos += 1
        while count == 0:
            if boxTaken[boxPos]:
                boxPos += 1
            else:
                count -= 1
        boxes[boxPos] = baseBoxes[ind[0]]
        boxTaken[boxPos] = True
        errArr = np.delete(np.delete(errArr, ind[0], axis=0), ind[1], axis=1)
        del baseBoxes[ind[0]]
        print(boxTaken)
    return boxes


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
    boxes = findBoxes(im)
    box1,err1 = findBestBox(im, boxes, char1)
    box2,err2 = findBestBox(im, boxes, char2)
    print(box1)
    print(box2)
    #box1 = boxes[0]
    #box2 = boxes[1]
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

def main2():
    im = Image.open("../testFiles/4e91.png")
    im2 = Image.open("../testFiles/4e8c.png")
    arr = np.logical_not(np.asarray(im2)).astype(int)
    bb = findBoundingBox(arr,1)
    char = im2.crop(bb.getBoxOutline())
    boxes = findBoxes(im)
    box = findBestBox(im, boxes, char)
    cut = im.crop(box.getBoxOutline())
    cut.save("../testFiles/cut.png")

if __name__ == "__main__":
    if (len(sys.argv) == 4 ):
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main2()