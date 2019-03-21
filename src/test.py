from PIL import Image
from layouts import applyPreciseLayout
from findPosition_v3 import findBestBoxes
import time

def main():
    im = Image.open("../testFiles/4ead.png")
    im1 = Image.open("../testFiles/4ea0.png")
    im2 = Image.open("../testFiles/53e3.png")
    im3 = Image.open("../testFiles/5196.png")
    im4 = Image.open("../testFiles/4e01.png")
    comps = [im1, im2, im3, im4]

    (boxes, error) = findBestBoxes(im, comps)

    out = applyPreciseLayout(boxes, comps)
    out.save("../testFiles/out.png")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")