from PIL import Image, ImageChops
import sys
import os
import csv
from layouts import layouts, applyLayout, BoundingBox, applyPreciseLayout

# global variables
BASE_IMAGES_PATH = ""
TARGET_PATH = ""
CHARACTER_DATABASE_PATH = ""

CHARDEF_DIC = {}
CHAR_DIC = {}

BASE_COUNT = 0
ROUGH_DEF_COUNT = 0
PRECISE_DEF_COUNT = 0

class CharDef:
    def __init__(self, lid, compIds, preciseDef=False, boxes=None):
        self.lid = lid
        self.compIds = compIds
        self.preciseDef = preciseDef
        self.boxes = boxes

def createCharDefDic():
    print("Creating character definition dictionary")
    global CHARDEF_DIC
    global BASE_COUNT
    global ROUGH_DEF_COUNT
    global PRECISE_DEF_COUNT
    with open(CHARACTER_DATABASE_PATH) as csvfile:
        fileReader = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in fileReader:
            count += 1
            Id = row[0]
            lid = int(row[1])
            compsLenght = len(layouts.get(lid))
            compIds = []
            for i in range(compsLenght):
                compIds.append(row[i + 2])
            preciseDef = int(row[compsLenght + 2])
            if preciseDef == 1:
                boxes = []
                for i in range(compsLenght):
                    x = int(row[1 + 4*i + 2 + compsLenght])
                    y = int(row[2 + 4*i + 2 + compsLenght])
                    dx = int(row[3 + 4*i + 2 + compsLenght])
                    dy = int(row[4 + 4*i + 2 + compsLenght])
                    box = BoundingBox(x,y,dx,dy)
                    boxes.append(box)
                PRECISE_DEF_COUNT += 1
                charDef = CharDef(lid, compIds, True, boxes)
            else:
                if lid == 0:
                    BASE_COUNT += 1
                else:
                    ROUGH_DEF_COUNT += 1
                charDef = CharDef(lid, compIds)
            CHARDEF_DIC.update({Id: charDef})
    print("Amount of base characters: ", BASE_COUNT)
    print("Amount of roughly defined characters: ", ROUGH_DEF_COUNT)
    print("Amount of precisely defined characters: ", PRECISE_DEF_COUNT)

def addBaseImages():
    print("Adding base images to dictionary")
    global CHAR_DIC
    for file in os.listdir(BASE_IMAGES_PATH):
        if file.endswith(".png"):
            Id = file[:-4]
            charDef = CHARDEF_DIC.get(Id)
            try:
                if charDef.lid == 0:
                    image = Image.open(BASE_IMAGES_PATH + file)
                    image = image.convert('1')
                    CHAR_DIC.update({Id: image})
            except Exception as error:
                pass

def createCharImage(Id):
    global CHAR_DIC
    try:
        cDef = CHARDEF_DIC.get(Id)
        lid = cDef.lid
        compIds = cDef.compIds
        preciseDef = cDef.preciseDef
        boxes = cDef.boxes
    except Exception as error:
        print(error)
    if lid == 0:
        raise Exception("Base image for character id " + Id + " was not provided")
    compImgs = []
    for i in compIds:
        if i not in CHAR_DIC.keys():
            createCharImage(i)
        img = CHAR_DIC.get(i)
        compImgs.append(img)
    if preciseDef:
        im = applyPreciseLayout(boxes, compImgs)
    else:
        im = applyLayout(lid,compImgs)
    CHAR_DIC.update({Id: im})

def completeImageDic():
    print("Creating new character images")
    global CHAR_DIC
    i = 0
    tot = len(CHARDEF_DIC)
    for key in CHARDEF_DIC.keys():
        i += 1
        perc = i*100//tot
        sys.stdout.write("\r%i %%" % perc)
        sys.stdout.flush()
        try:
            if key not in CHAR_DIC.keys():
                createCharImage(key)
        except Exception as error:
            print("")
            print("Error for key n°: ", key)
            print("Error message: ", error)
    print("")

def saveImages():
    print("Saving newly created images to output directory")
    for Id, im in CHAR_DIC.items():
        charDef = CHARDEF_DIC.get(Id)
        if charDef.lid != 0:
            if charDef.preciseDef:
                im.save(TARGET_PATH + "preciseDef/" + Id + ".png")
            else:
                im.save(TARGET_PATH + "roughDef/" + Id + ".png")


def createCharacterSet():
    createCharDefDic()
    addBaseImages()
    completeImageDic()
    saveImages()
    print("Done")

def main():
    if len(sys.argv) != 4:
        print("Wrong number of arguments, please provide:")
        print("1) a path to the directory containing the images of the base characters")
        print("2) a path to the target directory")
        print("3) the filename (with path) of the character database")
    else:
        global BASE_IMAGES_PATH
        global TARGET_PATH
        global CHARACTER_DATABASE_PATH
        BASE_IMAGES_PATH = sys.argv[1]
        TARGET_PATH = sys.argv[2]
        CHARACTER_DATABASE_PATH = sys.argv[3]
        createCharacterSet()

if __name__ == "__main__":
    main()