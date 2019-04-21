# -*- coding: utf-8 -*-

UTF_20902 = []
UNICODES = []

def prepareCharacters():
    global UTF_20902
    global UNICODES
    for i in range(0x4E00, 0x9FA5+1): 
        UTF_20902.append(chr(i))
        UNICODES.append(takeUnicode(chr(i)))


def takeUnicode(elem):
    return elem.encode('unicode_escape').decode()[2:]

def isValidValue(val):
    if val in UTF_20902 or val in UNICODES:
        return True
    return False

def handleValue(val):
    print("------------------")
    if val in UTF_20902:
        i = UTF_20902.index(val)
        print("Unicode for character " + val + " is: ", UNICODES[i])
    elif val in UNICODES:
        i = UNICODES.index(val)
        print("Character for unicode " + val + " is: ", UTF_20902[i])
    print("------------------")

def main():
    prepareCharacters()
    print_header = True
    buffer = ""
    while True:
        if (print_header):
            print("Enter the a unicode or a character for translation.")
            print("Enter a blank line to exit.")
        else:
            print_header = True
        line = input()
        if line == "":
            break
        buffer = line
        if isValidValue(buffer):
            handleValue(buffer)
        else:
            print_header = False
            print("Incorrect value.")

if __name__ == '__main__':
    main()