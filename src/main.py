from PIL import Image, ImageChops
import sys
import os
import csv
from class_definitions import BoundingBox
from layouts import layouts, applyLayout, applyPreciseLayout
import sqlite3
from sqlite_database import openDatabase, closeDatabase

# global variables
DATABASE_PATH = "../database.db"
BASE_IMAGES_PATH = ""
TARGET_PATH = ""

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
    conn, cur = openDatabase()
    chars = cur.execute("Select uid, lid, compsLen, comp1, comp2, comp3, comp4, comp5, pdef from Character").fetchall()
    for char in chars:
        uid = char[0]
        lid = char[1]
        compsLenght = char[2]
        compIds = []
        for i in range(compsLenght):
            compIds.append(char[3+i])
        preciseDef = char[8] != None
        if preciseDef:
            pdef = cur.execute("Select boxId1, boxId2, boxId3, boxId4, boxId5 from PreciseDef where pdefid = ?", (char[8],)).fetchone()
            boxes = []
            for i in range(compsLenght):
                (x,y,dx,dy) = cur.execute("Select x, y, dx, dy from Box where boxId = ?", (pdef[i],)).fetchone()
                boxes.append(BoundingBox(x,y,dx,dy))
            PRECISE_DEF_COUNT += 1
            charDef = CharDef(lid, compIds, True, boxes)
        else:
            if lid == 0:
                BASE_COUNT += 1
            else:
                ROUGH_DEF_COUNT += 1
            charDef = CharDef(lid, compIds)
        CHARDEF_DIC.update({uid: charDef})
    closeDatabase(conn)
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

def saveLossMapsAsImages():
    out_file = "../outputFiles/lossMaps/"
    conn, cur = openDatabase()
    uidAndLossMaps = cur.execute("Select uid, lossMap from LossMap").fetchall()
    for ualm in uidAndLossMaps:
        uid = ualm[0]
        arr = ualm[1].astype(float)
        arr = arr / np.amax(arr)
        arr = (arr * 255).astype(np.uint8)
        im = Image.fromarray(arr, mode='L')
        im.save(out_file + str(uid) + ".png")


def createCharacterSet():
    createCharDefDic()
    addBaseImages()
    completeImageDic()
    saveImages()
    #saveLossMapsAsImages()
    print("Done")

def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments, please provide:")
        print("1) a path to the directory containing the images of the base characters")
        print("2) a path to the target directory")
    else:
        global BASE_IMAGES_PATH
        global TARGET_PATH
        global CHARACTER_DATABASE_PATH
        BASE_IMAGES_PATH = sys.argv[1]
        TARGET_PATH = sys.argv[2]
        createCharacterSet()

if __name__ == "__main__":
    main()