import time
import numpy as np
from PIL import Image
from array_display import display_array

def main():
    line = input()
    key = line.split(" ")
    print(key)
    if len(key) == 2:
        uid = key[0]
        style = key[1]
        print((uid, style))
        print((style, uid))

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution time: " + str(int(end-start)) + " seconds")