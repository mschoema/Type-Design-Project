from PIL import Image
from layouts import applyPreciseLayout
from findPosition_v2 import findBoxes, matrixElim, findBestBox
from findPosition_v3 import findBestBoxes
import time

def main():
    im = Image.open("../CharacterImages/5669.png")
    im1 = Image.open("../CharacterImages/738b.png")
    im2 = Image.open("../CharacterImages/53e3.png")
    im3 = Image.open("../CharacterImages/53e3.png")
    im4 = Image.open("../CharacterImages/53e3.png")
    im5 = Image.open("../CharacterImages/53e3.png")
    comps = [im1, im2, im3, im4, im5]

    baseBoxes = findBoxes(im)

    if len(baseBoxes) == len(comps):
        print("Ok")
        boxes = matrixElim(im, baseBoxes, comps)
    else:
        print("Not Ok")
        boxes = []
        for comp in comps:
            (box, error) = findBestBox(im, baseBoxes, comp)
            boxes.append(box)

    print(boxes)

    out = applyPreciseLayout(boxes, comps)
    out.save("../testFiles/out.png")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")