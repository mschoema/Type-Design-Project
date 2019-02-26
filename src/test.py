from PIL import Image, ImageChops
import sys
import os
import csv
from layouts import layouts, applyLayout

# global variables
BASE_IMAGES_PATH = ""
TARGET_PATH = ""
CHARACTER_DATABASE_PATH = ""

CHARDEF_DIC = {}
CHAR_DIC = {}

class CharDef:
    def __init__(self, lid, compIds):
        self.lid = lid
        self.compIds = compIds

def addBaseImages():
    global CHAR_DIC
    for file in os.listdir(BASE_IMAGES_PATH):
        if file.endswith(".png"):
            Id = file[:-4]
            image = Image.open(BASE_IMAGES_PATH + file)
            image = image.convert('1')
            CHAR_DIC.update({Id: image})

def addBaseImages2():
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
                print(Id)


def createCharDefDic():
    global CHARDEF_DIC
    with open(CHARACTER_DATABASE_PATH) as csvfile:
        fileReader = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in fileReader:
            # TODO: verify numbers
            count += 1
            Id = row[0]
            lid = int(row[1])
            compsLenght = len(layouts.get(lid))
            compIds = []
            for i in range(compsLenght):
                compIds.append(row[i + 2])
            charDef = CharDef(lid, compIds)
            CHARDEF_DIC.update({Id: charDef})

def createCharImage(Id):
    global CHAR_DIC
    try:
        cDef = CHARDEF_DIC.get(Id)
    except Exception as error:
        print(error)
    lid = cDef.lid
    compIds = cDef.compIds
    if lid == 0:
        raise Exception("Base image for character id " + Id + " was not provided.")
    compImgs = []
    for i in compIds:
        if i not in CHAR_DIC.keys():
            createCharImage(i)
        img = CHAR_DIC.get(i)
        compImgs.append(img)
    im = applyLayout(lid,compImgs)
    CHAR_DIC.update({Id: im})

def completeImageDic():
    global CHAR_DIC
    try:
        for key in CHARDEF_DIC.keys():
            if key not in CHAR_DIC.keys():
                createCharImage(key)
    except Exception as error:
        print(error)

def saveImages():
    for Id, im in CHAR_DIC.items():
        charDef = CHARDEF_DIC.get(Id)
        if charDef.lid != 0:
            im.save(TARGET_PATH + Id + ".png")


def createCharacterSet():
    createCharDefDic()
    #addBaseImages()
    addBaseImages2()
    completeImageDic()
    saveImages()

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