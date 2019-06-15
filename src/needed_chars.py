from sqlite_database import openDatabase, closeDatabase
from character_to_unicode import getChar

def main():
    (conn, cur) = openDatabase()

    unicodes = cur.execute("Select uid from Character where lid == 0").fetchall()
    unicodes_list = [uid[0] for uid in unicodes]

    chars_needed = ""
    for uid in unicodes_list:
        char = getChar(uid)
        chars_needed += char

    closeDatabase(conn)

    print(chars_needed)

if __name__ == '__main__':
    main()