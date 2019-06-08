from PIL import Image
import os.path
import glob
import os
import string
import argparse

UNIT_SIZE = 256
TARGET_WIDTH = 2 * UNIT_SIZE

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--inPath', type=str, default = "../outputFiles/667a/rough/")
    parser.add_argument('--trainPath', type=str, default = "../outputFiles/train_667a/rough/")

    args = parser.parse_args()
    inPath = args.inPath
    trainPath = args.trainPath
    train_dir = os.path.dirname(trainPath)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    images = []
    for f in os.listdir(inPath):
        if (f != '.DS_Store'):
            images.append(f)

    # make the training dataset
    for i in range(len(images)):
        im = Image.open(os.path.join(inPath,images[i]))
        target = im.resize((TARGET_WIDTH, UNIT_SIZE))
        quality_value = 100
        target.save(trainPath+'0_'+str(i).zfill(4)+'_'+"1"+'.png', quality = quality_value)