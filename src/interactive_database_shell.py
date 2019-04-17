import re
import sqlite_database as db
from class_definitions import CharDef, BoundingBox

DATABASE_PATH = "../database.db"

UNICODES = None
MODIFIED_CHARS = {}
NEW_CHARS = {}

def setupUnicodeList():
    global UNICODES
    UNICODES = db.getUnicodeList()

def isValidUnicode(buffer):
    return re.match('[0-9][0-9a-f]{3}', buffer)

def printCharDef(charDef):
    print("------------------")
    print("Character: ", charDef.uid)
    print("Base layout:  ", charDef.lid)
    print("Components: ", charDef.compIds)
    if charDef.preciseDef:
        print("Character has a precise definition")
        print("Boxes: ",charDef.boxes)
    else:
        print("Character does not have a precise definition.")
    print("------------------")

def getSafeInt(str):
    if str == "":
        raise Exception("Cancelling.")
    try:
        i = int(str)
    except Exception as e:
        raise Exception("Value provided was not a number, please start over.")
    return i

def addPreciseDefinition(charDef):
    boxes = []
    print("------------------")
    for i in range(charDef.compsLen):
        print("Box for component " + str(i+1) + " (character unicode: " + str(charDef.compIds[i]) + ")")
        print("Enter a random letter to cancel")
        x = getSafeInt(input('x = '))
        y = getSafeInt(input('y = '))
        dx = getSafeInt(input('dx = '))
        dy = getSafeInt(input('dy = '))
        box = BoundingBox(x,y,dx,dy)
        boxes.append(box)
    charDef.setPreciseDef(boxes)
    print("------------------")
    return charDef

def handleUnicode(uid):
    if uid in UNICODES:
        charDef = db.getCharDef(uid)
        printCharDef(charDef)
        print_header = True
        buffer = ""
        while True:
            if (print_header):
                print("What do you want to do with this character")
                print("Enter 1 to add or change a precise definition.")
                print("Enter a blank line to choose another character.")
            else:
                print_header = True
            line = input()
            if line == "":
                break
            buffer = line
            if buffer == "1":
                try:
                    charDef = addPreciseDefinition(charDef)
                    db.addPreciseDef(charDef)
                    print("Precise definition successfully added to the database")
                    printCharDef(charDef)
                except Exception as e:
                    print(e)
                    print("------------------")
            else:
                print_header = False
                print("Incorrect value.")
    else:
        print(":(")

def main():
    setupUnicodeList()
    print_header = True
    buffer = ""
    while True:
        if (print_header):
            print("Enter the unicode of a character.")
            print("Enter a blank line to exit.")
        else:
            print_header = True
        line = input()
        if line == "":
            break
        buffer = line
        if isValidUnicode(buffer):
            handleUnicode(buffer)
        else:
            print_header = False
            print("Incorrect unicode.")

if __name__ == '__main__':
    main()
