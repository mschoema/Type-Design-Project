import sys
import csv
from layouts import layouts

UNICODE_DIC = {}
CHAR_DIC = {}
READABLE_CHAR_DIC = {}
CHARACTER_DATABASE_PATH = ""
OUTPUT_DATABASE_PATH = ""

class CharDef:
    def __init__(self, lid, compChars):
        self.lid = lid
        self.compChars = compChars

class IdCharDef:
    def __init__(self, lid, compIds):
        self.lid = lid
        self.compIds = compIds

def getCompIds(compChars):
    global UNICODE_DIC
    compIds = []
    for char in compChars:
        Id = UNICODE_DIC.get(char)
        compIds.append(Id)
    return compIds

def createDatabaseCSV():
    global CHAR_DIC
    global UNICODE_DIC
    global READABLE_CHAR_DIC
    with open(CHARACTER_DATABASE_PATH, encoding="utf16") as csvfile:
        fileReader = csv.reader(csvfile, delimiter='\t')
        count = 0
        for row in fileReader:
            if count == 0:
                count += 1
                continue
            try:
                lid = int(row[2][3:])
            except Exception as error:
                break
            print(count)
            count += 1
            char = row[0]
            Id = row[1]
            compsLenght = len(layouts.get(lid))
            compChars = []
            for i in range(compsLenght):
                compChars.append(row[i + 3])
            charDef = CharDef(lid, compChars)
            UNICODE_DIC.update({char: Id})
            CHAR_DIC.update({Id: charDef})
    for k,v in CHAR_DIC.items():
        comIds = getCompIds(v.compChars)
        charDef = IdCharDef(v.lid, comIds)
        READABLE_CHAR_DIC.update({k: charDef})
    with open(OUTPUT_DATABASE_PATH + "database.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for k,v in READABLE_CHAR_DIC.items():
            toWrite = []
            toWrite.append(k)
            none = False
            toWrite.append(v.lid)
            for Id in v.compIds:
                if Id == None:
                    print("None")
                    none = True
                toWrite.append(Id)
            #toWrite.append(v.compIds)
            if none:
                writer.writerow([k, 0, k])
            else:
                writer.writerow(toWrite)


    

def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments, please provide:")
        print("1) the filename (with path) of the character database")
        print("2) the path for the newly created character database")
    else:
        global CHARACTER_DATABASE_PATH
        global OUTPUT_DATABASE_PATH
        CHARACTER_DATABASE_PATH = sys.argv[1]
        OUTPUT_DATABASE_PATH = sys.argv[2]
        createDatabaseCSV()

if __name__ == "__main__":
    main()