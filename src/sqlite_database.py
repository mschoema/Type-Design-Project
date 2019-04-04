import sqlite3
import numpy as np
import io
import csv
from PIL import Image
from layouts import layouts, BoundingBox
from loss_map import compute_loss_map
from array_display import display_array

class CharDef:
    def __init__(self, uid, lid, compsLength, compIds, preciseDef=False, boxes=None):
        self.uid = uid
        self.lid = lid
        self.compsLen = compsLength
        self.compIds = compIds
        self.preciseDef = preciseDef
        self.boxes = boxes

"""
Converting numpy array to binary and back for insertion in sql
"""

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

DATABASE_PATH = "../database.db"
CHARCTER_IMAGES_PATH = "../CharacterImages/Regular/"
CREATE_DATABASE = False
FILL_DATABASE = False
COMPUTE_LOSS_MAPS = False
SHOW_LOSS_MAPS = True
SHELL = False

def openDatabase():
    conn = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()
    return (conn,cur)

def closeDatabase(conn):
    conn.commit()
    conn.close()

def createDatabase():
    print("Creating database")
    (conn,cur) = openDatabase()
    cur.executescript("""
        CREATE table if not exists Character(
            uid Text primary key,
            lid Integer not null,
            compsLen Integer not null,
            comp1 Text not null,
            comp2 Text,
            comp3 Text,
            comp4 Text,
            comp5 Text,
            pdef Integer,
            Foreign key (pdef) references PreciseDef(pdefid)
        );

        CREATE table if not exists PreciseDef(
            pdefid Integer primary key,
            boxId1 Integer not null,
            boxId2 Integer not null,
            boxId3 Integer,
            boxId4 Integer,
            boxId5 Integer,
            Foreign key (boxId1) references Box(boxId),
            Foreign key (boxId2) references Box(boxId),
            Foreign key (boxId3) references Box(boxId),
            Foreign key (boxId4) references Box(boxId),
            Foreign key (boxId5) references Box(boxId)
        );

        CREATE table if not exists Box(
            boxId Integer primary key,
            x Integer not null,
            y Integer not null,
            dx Integer not null,
            dy Integer not null
        );

        CREATE table if not exists LossMap(
            uid Text primary key,
            lossMap array,
            Foreign key (uid) references Character(uid)
        );

        """)
    closeDatabase(conn)
    print("Done")

def fillDatabase():
    print("Filling database")

    def insertPreciseDef(charDef):
        if charDef.preciseDef:
            boxIds = []
            for i in range(5):
                if i < charDef.compsLen:
                    box = charDef.boxes[i]
                    cur.execute("Insert Into Box(x,y,dx,dy) values (?,?,?,?)", (box.x, box.y, box.dx, box.dy))
                    boxIds.append(cur.lastrowid)
                else:
                    boxIds.append(None)
            cur.execute("Insert Into PreciseDef(boxId1, boxId2, boxId3, boxId4, boxId5) values (?,?,?,?,?)",boxIds)
            return cur.lastrowid
        else:
            return None

    def insertChar(charDef):
        pdefid = insertPreciseDef(charDef)
        char = [charDef.uid, charDef.lid, charDef.compsLen]
        for i in range(5):
            if i < charDef.compsLen:
                char.append(charDef.compIds[i])
            else:
                char.append(None)
        char.append(pdefid)
        cur.execute("Insert Into Character(uid,lid,compsLen, comp1, comp2, comp3, comp4, comp5, pdef) values (?,?,?,?,?,?,?,?,?)", char)

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
            insertChar(charDef)
    closeDatabase(conn)
    print("Done")

def computeLossMaps():
    print("Computing loss maps")
    (conn, _) = openDatabase()
    conn.row_factory = lambda cursor, row: row[0]
    cur = conn.cursor()
    uids = cur.execute("Select uid from Character where lid != 0").fetchall()
    tot = len(uids)
    print(str(tot) + " loss maps to compute")
    for uid in uids:
        im = Image.open(CHARCTER_IMAGES_PATH + str(uid) + ".png")
        im = im.convert('1')
        #im = im.resize((100, 100))
        arr = np.logical_not(np.asarray(im)).astype(int)
        lossMap = compute_loss_map(arr)
        cur.execute("Insert Into LossMap(uid,lossMap) values (?,?)", (uid, lossMap))
    closeDatabase(conn)
    print("Done")

def showLossMaps():
    (conn, _) = openDatabase()
    uidAndLossMaps = cur.execute("Select uid, lossMap from LossMap").fetchall()
    print(uidAndLossMaps)
    uids = []
    lossMaps = []
    closeDatabase(conn)
    print("Enter a unicode to see the loss map.")
    print("Enter a blank line to exit.")
    while True:
        line = input()
        if line == "":
            break
        if line in uids:
            index = uids.index(line)
            display_array(lossMaps[index])
            print("---")
        else:
            print("Wrong unicode")

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
    if CREATE_DATABASE:
        createDatabase()
    if FILL_DATABASE:
        fillDatabase()
    if COMPUTE_LOSS_MAPS:
        computeLossMaps()
    if SHOW_LOSS_MAPS:
        showLossMaps()
    if SHELL:
        sqlShell()

if __name__ == '__main__':
    main()