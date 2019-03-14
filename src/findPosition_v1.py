import numpy as np
from PIL import Image, ImageChops, ImageStat
from layouts import findBoundingBox, BoundingBox
import sys

def findPosition(image, char):
    print("Looking for position")
    arr2 = np.asarray(image)
    maxWidth, maxHeigth = char.size
    minWidth, minHeigth = maxWidth//3, maxHeigth//3
    minError = 99999999
    bestBox = None
    i = 0
    print("amount of sizes to try: ", ((maxWidth - minWidth)//25)*((maxHeigth - minHeigth)//25))
    for w in range(maxWidth, minWidth, -120):
        for h in range(maxHeigth, minHeigth, -120):
            i += 1
            sys.stdout.write("\rtry n° %i" % i)
            sys.stdout.flush()
            resIm = char.resize((w,h))
            maxX = 1000-w
            maxY = 1000-h
            for x in range(0,maxX,25):
                for y in range(0,maxY,25):
                    boxVerif = Image.new("1",(1000,1000),1)
                    boxVerif.paste(resIm,(x,y))
                    arr1 = np.asarray(boxVerif)
                    arrOr = np.logical_or(arr1,arr2)
                    arrXor = np.logical_xor(arr1,arrOr)
                    error = np.count_nonzero(arrXor)/(w*h)
                    if error < minError:
                        print("Found better: ", error)
                        minError = error
                        bestBox = BoundingBox(x,y,w,h)
    return bestBox

def main():
    im1 = Image.open("../testFiles/4eba.png")
    im2 = Image.open("../testFiles/6b62.png")
    im3 = Image.open("../testFiles/4f01.png")
    bb1 = findBoundingBox(im1)
    bb2 = findBoundingBox(im2)
    print(bb1)
    print(bb2)
    crp1 = im1.crop(bb1.getBoxOutline())
    crp2 = im2.crop(bb2.getBoxOutline())
    pos1 = findPosition(im3, crp1)
    pos2 = findPosition(im3, crp2)
    cut1 = im3.crop(pos1.getBoxOutline())
    cut2 = im3.crop(pos2.getBoxOutline())
    cut1.save("../testFiles/cut1.png")
    cut2.save("../testFiles/cut2.png")
    char1 = crp1.resize(pos1.getSize())
    char2 = crp2.resize(pos2.getSize())
    out = Image.new("1",(1000,1000),1)
    out.paste(char1, pos1.getBoxOutline(), ImageChops.invert(char1))
    out.paste(char2, pos2.getBoxOutline(), ImageChops.invert(char2))
    out.save("../testFiles/out.png")

def main2():
    im = Image.open("../testFiles/4e91.png")
    im2 = Image.open("../testFiles/4e8c.png")
    bb = findBoundingBox(im2)
    char = im2.crop(bb.getBoxOutline())
    box = findPosition(im, char)
    cut = im.crop(box.getBoxOutline())
    cut.save("../testFiles/cut.png")
    crp = char.resize(box.getSize())
    im3 = im.copy()
    im3.paste(crp, box.getBoxOutline())
    im3.save("../testFiles/out.png")

if __name__ == "__main__":
    main2()