# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import csv
import sys
import os
from PIL import Image
from class_definitions import BoundingBox, CharDef, ArrayCollection
from layouts import layouts, applyLayout, applyPreciseLayout
from array_display import display_array
from loss_map import compute_loss_map
import sqlite_database
from sqlite_database import openDatabase, closeDatabase
import copy

"""
Converting numpy array to binary and back for insertion in sql
"""

DATABASE_PATH = "../database.db"
TEXT_DATABASE_PATH = "../inputFiles/database4.0.txt"
CHARACTER_IMAGES_PATH = "../characterImages/"
OUTPUT_PATH = "../outputFiles/"
FILL_DATABASE = False
UPDATE_DATABASE = False
COMPUTE_AND_IMPORT_ARRAYS = True
SHOW_ARRAYS = False
SHELL = False
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

def fillDatabase():
    print("Filling database")
    path_to_csv = "../inputFiles/database.csv"
    (conn,cur) = openDatabase()
    with open(path_to_csv) as csvfile:
        fileReader = csv.reader(csvfile, delimiter=',')
        for row in fileReader:
            uid = row[0]
            lid = int(row[1])
            compsLength = len(layouts.get(lid))
            compIds = []
            for i in range(compsLength):
                compIds.append(row[i + 2])
            preciseDef = int(row[compsLength + 2])
            if preciseDef == 1:
                boxes = []
                for i in range(compsLength):
                    x = int(row[1 + 4*i + 2 + compsLength])
                    y = int(row[2 + 4*i + 2 + compsLength])
                    dx = int(row[3 + 4*i + 2 + compsLength])
                    dy = int(row[4 + 4*i + 2 + compsLength])
                    box = BoundingBox(x,y,dx,dy)
                    boxes.append(box)
                charDef = CharDef(uid, lid, compsLength, compIds, True, boxes)
            else:
                charDef = CharDef(uid, lid, compsLength, compIds)
            sqlite_database.insertChar(charDef)
    closeDatabase(conn)
    print("Done")

def updateDatabase(path_to_database, path_to_char_to_uid):

    def getCharToUid():
        char_to_uid_dic = {}
        with open(path_to_char_to_uid, encoding="utf16") as csvfile:
            fileReader = csv.reader(csvfile, delimiter='\t')
            count = 0
            for row in fileReader:
                if count == 0:
                    count += 1
                    continue
                char_to_uid_dic.update({row[0]: row[1]})
        return char_to_uid_dic

    def getUid(char):
        return char_to_uid_dic.get(char)

    def readRow(row):
        try:
            uid = getUid(row[0])
            lid = int(row[1][3:])
            compsLength = len(layouts.get(lid))
            compIds = []
            for i in range(compsLength):
                compIds.append(getUid(row[i + 2]))
            charDef = CharDef(uid, lid, compsLength, compIds)
        except Exception as e:
            print(e)
            charDef = None
        return charDef

    print("Updating database")
    characters = sqlite_database.getUnicodeList()
    waitingList = []
    char_to_uid_dic = getCharToUid()
    with open(path_to_database, encoding="utf16") as csvfile:
        fileReader = csv.reader(csvfile, delimiter='\t')
        count = 0
        for row in fileReader:
            if count == 0:
                count += 1
                continue
            count += 1
            charDef = readRow(row)
            print(charDef)
            if charDef:
                waitingList = [charDef] + waitingList
    while len(waitingList) > 0:
        newWaitingList = []
        beforeLen = len(waitingList)
        for i in reversed(range(beforeLen)):
            charDef = waitingList[i]
            if charDef.uid in characters:
                charDefInDb = sqlite_database.getCharDef(charDef.uid)
                if charDefInDb.lid == charDef.lid and charDefInDb.compIds == charDef.compIds:
                    pass
                else:
                    compsInDb = True
                    for i in range(charDef.compsLen):
                        if charDef.compIds[i] not in characters:
                            compsInDb = False
                    if compsInDb or charDef.lid == 0:
                        sqlite_database.updateChar(charDef)
                    else:
                        newWaitingList.append(charDef)
            else:
                compsInDb = True
                for i in range(charDef.compsLen):
                    if charDef.compIds[i] not in characters:
                        compsInDb = False
                if compsInDb or charDef.lid == 0:
                    if charDef.uid == "765a":
                        print("Hello1")
                    sqlite_database.insertChar(charDef)
                    characters.append(charDef.uid)
                else:
                    newWaitingList.append(charDef)
                if charDef.uid == "765a":
                    print("Hello2")
        afterLen = len(newWaitingList)
        if beforeLen == afterLen:
            break
        else:
            waitingList = newWaitingList
    if len(waitingList) > 0:
        print(len(waitingList))
        print("The following characters could not be inserted in the database:")
        for charDef in waitingList:
            print(charDef)
    print("Done")

def computeAndImportArrays(style):

    def getBaseImages(style):
        print("Adding base images to dictionary")
        roughImageDic = {}
        arrayDic = {}
        for file in os.listdir(CHARACTER_IMAGES_PATH + style + "/"):
            if file.endswith(".png"):
                uid = file[:-4]
                charDef = charDefDic.get(uid)
                if charDef != None:
                    try:
                        image = Image.open(CHARACTER_IMAGES_PATH + style + "/" + file)
                        image = image.convert('1')
                        image = image.resize(IMAGE_SIZE)
                        if charDef.lid == 0:
                            roughImageDic.update({uid: image})
                        else:
                            # arr = np.logical_not(np.asarray(image)).astype(int)
                            # ac = ArrayCollection(uid, style, arr)
                            ac = ArrayCollection(uid, style, image)
                            arrayDic.update({uid: ac})
                    except Exception as error:
                        print("Error encountered for unicode: ",uid)
                        print("Error message: ", error)
        return roughImageDic, arrayDic

    def createCharImage(uid):
        try:
            cDef = charDefDic.get(uid)
            lid = cDef.lid
            compIds = cDef.compIds
            preciseDef = cDef.preciseDef
            boxes1000 = cDef.boxes 
        except Exception as error:
            print("Error encountered for unicode: ",charDef.uid)
            print("Error message: ", error)
        if lid == 0:
            raise Exception("Base image for character id " + uid + " was not provided")
        compImgs = []
        for i in compIds:
            if i not in roughImageDic.keys():
                createCharImage(i)
            img = roughImageDic.get(i)
            compImgs.append(img)
        if preciseDef:
            boxes = []
            for box1000 in boxes1000:
                x = int(box1000.x*IMAGE_WIDTH/1000)
                y = int(box1000.y*IMAGE_HEIGHT/1000)
                dx = int(box1000.dx*IMAGE_HEIGHT/1000)
                dy = int(box1000.dy*IMAGE_HEIGHT/1000)
                box = BoundingBox(x,y,dx,dy)
                boxes.append(box)
            im = applyPreciseLayout(boxes, compImgs)
        else:
            im = applyLayout(lid,compImgs)
        roughImageDic.update({uid: im})

    def addRoughDefArrays():
        print("Creating new character images")
        i = 0
        tot = len(charDefDic)
        for uid, charDef in charDefDic.items():
            i += 1
            perc = i*100//tot
            sys.stdout.write("\r%i %%" % perc)
            sys.stdout.flush()
            if charDef.lid != 0:
                try:
                    if uid not in roughImageDic.keys():
                        createCharImage(uid)
                    roughImage = roughImageDic.get(uid)
                    # arr = np.logical_not(np.asarray(roughImage)).astype(int)
                    ac = arrayDic.get(uid)
                    # ac.addRoughDef(arr)
                    ac.addRoughDef(roughImage)
                except Exception as error:
                        print("Error encountered for unicode: ",uid)
                        print("Error message: ", error)
        print("")


    def addLossMapArrays():
        print("Computing loss maps")
        i = 0
        tot = len(arrayDic)
        for uid, ac in arrayDic.items():
            i += 1
            perc = i*100//tot
            sys.stdout.write("\r%i %%" % perc)
            sys.stdout.flush()
            # arr = ac.character
            im = ac.character
            arr = np.logical_not(np.asarray(im)).astype(int)
            # lossMap = compute_loss_map(arr)
            arr = compute_loss_map(arr).astype(float)
            arr = arr / np.amax(arr)
            arr = (arr * 255).astype(np.uint8)
            lossMap = Image.fromarray(arr, mode='L')
            ac.addLossMap(lossMap)
        print("")

    print("Importing arrays for style: ", style)
    charDefDic = sqlite_database.getCharDefDic()
    roughImageDic, arrayDic = getBaseImages(style)
    addRoughDefArrays()
    # addLossMapArrays()
    print("Saving images")
    for uid, ac in arrayDic.items():
        if ac.isComplete():
            images = [ac.character, ac.roughDefinition, ac.lossMap]

            images[1].save(OUTPUT_PATH + "/rough_1000/" + ac.style + "_" + ac.uid + ".png")
            images[0].save(OUTPUT_PATH + "/origin_1000/" + ac.style + "_" + ac.uid + ".png")
            # images[2].save(OUTPUT_PATH + "/lossmap_1000/" + ac.style + "_" + ac.uid + ".png")

            images[1].save(OUTPUT_PATH + ac.style + "/rough_1000/" + ac.uid + ".png")
            images[0].save(OUTPUT_PATH + ac.style + "/origin_1000/" + ac.uid + ".png")
            # images[2].save(OUTPUT_PATH + ac.style + "/lossmap_1000/" + ac.uid + ".png")
        else:
            ac.printIncomplete()
    print("Done")

def sqlShell():
    (conn, cur) = openDatabase()
    buffer = ""
    print("Enter your SQL commands to execute in sqlite3.")
    print("Enter a blank line to exit.")
    while True:
        line = input()
        if line == "":
            break
        buffer += line
        if sqlite3.complete_statement(buffer):
            try:
                buffer = buffer.strip()
                cur.execute(buffer)

                if buffer.lstrip().upper().startswith("SELECT"):
                    print(cur.fetchall())
            except sqlite3.Error as e:
                print("An error occurred:", e.args[0])
            buffer = ""
        else:
            print("Wrong statement")
    closeDatabase(conn)

def main():
    if FILL_DATABASE:
        fillDatabase()
    if UPDATE_DATABASE:
        path_to_database = sys.argv[1]
        path_to_char_to_uid = sys.argv[2]
        updateDatabase(path_to_database, path_to_char_to_uid)
    if COMPUTE_AND_IMPORT_ARRAYS:
        styles = sys.argv[1:]
        rough_directory = os.path.dirname('../outputFiles/rough_1000/')
        origin_directory = os.path.dirname('../outputFiles/origin_1000/')
        lossmap_directory = os.path.dirname('../outputFiles/lossmap_1000/')
        if not os.path.exists(rough_directory):
            os.makedirs(rough_directory)
        if not os.path.exists(origin_directory):
            os.makedirs(origin_directory)
        if not os.path.exists(lossmap_directory):
            os.makedirs(lossmap_directory)
        for style in styles:
            rough_directory = os.path.dirname('../outputFiles/{}/rough_1000/'.format(style))
            origin_directory = os.path.dirname('../outputFiles/{}/origin_1000/'.format(style))
            lossmap_directory = os.path.dirname('../outputFiles/{}/lossmap_1000/'.format(style))
            if not os.path.exists(rough_directory):
                os.makedirs(rough_directory)
            if not os.path.exists(origin_directory):
                os.makedirs(origin_directory)
            if not os.path.exists(lossmap_directory):
                os.makedirs(lossmap_directory)
            computeAndImportArrays(style)
    if SHELL:
        sqlShell()

if __name__ == '__main__':
    main()
