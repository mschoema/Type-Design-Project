import sqlite3
import numpy as np
import io
from class_definitions import BoundingBox, CharDef, ArrayCollection

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
        """)
    closeDatabase(conn)
    print("Done")

def addPreciseDef(charDef):
    (conn, cur) = openDatabase()
    boxIds = []
    for i in range(5):
        if i < charDef.compsLen:
            box = charDef.boxes[i]
            cur.execute("Insert Into Box(x,y,dx,dy) values (?,?,?,?)", (box.x, box.y, box.dx, box.dy))
            boxIds.append(cur.lastrowid)
        else:
            boxIds.append(None)
    cur.execute("Insert Into PreciseDef(boxId1, boxId2, boxId3, boxId4, boxId5) values (?,?,?,?,?)",boxIds)
    pdefId = cur.lastrowid
    cur.execute("Update Character set pdef = ? where uid = ?", (pdefId, charDef.uid))
    closeDatabase(conn)

def insertPreciseDef(charDef, cur):
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
    (conn, cur) = openDatabase()
    pdefid = insertPreciseDef(charDef, cur)
    char = [charDef.uid, charDef.lid, charDef.compsLen]
    for i in range(5):
        if i < charDef.compsLen:
            char.append(charDef.compIds[i])
        else:
            char.append(None)
    char.append(pdefid)
    cur.execute("Insert Into Character(uid,lid,compsLen, comp1, comp2, comp3, comp4, comp5, pdef) values (?,?,?,?,?,?,?,?,?)", char)
    closeDatabase(conn)

def updateChar(charDef):
    (conn, cur) = openDatabase()
    char = [charDef.lid, charDef.compsLen]
    for i in range(5):
        if i < charDef.compsLen:
            char.append(charDef.compIds[i])
        else:
            char.append(None)
    char.append(charDef.uid)
    cur.execute("Update Character set lid = ?, compsLen = ?, comp1 = ?, comp2 = ?, comp3 = ?, comp4 = ?, comp5 = ? where uid = ?", char)
    closeDatabase(conn)


def getUnicodeList():
    (conn, cur) = openDatabase()
    unicodes = cur.execute("Select uid from Character").fetchall()
    closeDatabase(conn)
    return [uid[0] for uid in unicodes]

def getCharDef(uid):
    (conn, cur) = openDatabase()
    charDefAsList = cur.execute("Select lid, compsLen, comp1, comp2, comp3, comp4, comp5, pdef from Character where uid = ?", (uid,)).fetchone()
    lid = charDefAsList[0]
    compsLength = charDefAsList[1]
    compIds = []
    for i in range(compsLength):
            compIds.append(charDefAsList[2+i])
    preciseDef = charDefAsList[7] != None
    if preciseDef:
        pdef = cur.execute("Select boxId1, boxId2, boxId3, boxId4, boxId5 from PreciseDef where pdefid = ?", (charDefAsList[7],)).fetchone()
        boxes = []
        for i in range(compsLength):
            (x,y,dx,dy) = cur.execute("Select x, y, dx, dy from Box where boxId = ?", (pdef[i],)).fetchone()
            boxes.append(BoundingBox(x,y,dx,dy))
        charDef = CharDef(uid, lid, compsLength, compIds, True, boxes)
    else:
        charDef = CharDef(uid, lid, compsLength, compIds)
    closeDatabase(conn)
    return charDef

def getCharDefDic():
    charDefDic = {}
    bCount = 0
    rdCount = 0
    pdCount = 0
    (conn, cur) = openDatabase()
    chars = cur.execute("Select uid, lid, compsLen, comp1, comp2, comp3, comp4, comp5, pdef from Character").fetchall()
    for char in chars:
        uid = char[0]
        lid = char[1]
        compsLength = char[2]
        compIds = []
        for i in range(compsLength):
            compIds.append(char[3+i])
        preciseDef = char[8] != None
        if preciseDef:
            pdef = cur.execute("Select boxId1, boxId2, boxId3, boxId4, boxId5 from PreciseDef where pdefid = ?", (char[8],)).fetchone()
            boxes = []
            for i in range(compsLength):
                (x,y,dx,dy) = cur.execute("Select x, y, dx, dy from Box where boxId = ?", (pdef[i],)).fetchone()
                boxes.append(BoundingBox(x,y,dx,dy))
            pdCount += 1
            charDef = CharDef(uid, lid, compsLength, compIds, True, boxes)
        else:
            if lid == 0:
                bCount += 1
            else:
                rdCount += 1
            charDef = CharDef(uid, lid, compsLength, compIds)
        charDefDic.update({uid: charDef})
    closeDatabase(conn)
    #print("Amount of base characters: ", bCount)
    #print("Amount of roughly defined characters: ", rdCount)
    #print("Amount of precisely defined characters: ", pdCount)
    return charDefDic

def main():
    createDatabase()

if __name__ == '__main__':
    main()