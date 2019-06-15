from PIL import Image
import os.path
import glob
import os
import string
import argparse

UNIT_SIZE = 256
TARGET_WIDTH = 2 * UNIT_SIZE
SPECIAL_UIDS = []
# SPECIAL_UIDS = ["673a", "5668", "5b66", "4e60", 
#                 "4e2d", "6587", "5b57", "4f53", 
#                 "63a2", "7d22", "6280", "672f", 
#                 "8bbe", "8ba1", "672a", "6765"]

# convert the size of png to 256*256
def convertpng(pngfile,outdir,width=256,height=256):
    out_path = os.path.dirname(outdir)
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    img=Image.open(pngfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(pngfile)))
    except Exception as e:
        print(e)

# convert the size of png images in one directory
def convertAllpng(convertInPath, convertOutPath):
    for pngfile in glob.glob(convertInPath):
        convertpng(pngfile, convertOutPath)

# convert the type of imgaes to jpg
def convertImageType(dirName):
    li=os.listdir(dirName)
    for filename in li:
        newname = filename
        newname = newname.split(".")
        if newname[-1]=="png":
            newname[-1]="jpg"
            newname = str.join(".",newname)  
            filename = dirName+filename
            newname = dirName+newname
            os.rename(filename,newname)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--roughConvertInPath', type=str, default = "../outputFiles/Songti-for-training/rough_1000/")
    parser.add_argument('--roughConvertOutPath', type=str, default = "../outputFiles/Songti-for-training/rough_256/")
    parser.add_argument('--originConvertInPath', type=str, default = "../outputFiles/Songti-for-training/origin_1000/")
    parser.add_argument('--originConvertOutPath', type=str, default = "../outputFiles/Songti-for-training/origin_256/")
    parser.add_argument('--trainPath', type=str, default = "../outputFiles/Songti-for-training/train_256/")

    args = parser.parse_args()
    roughConvertInPath = args.roughConvertInPath+"*.png"
    roughConvertOutPath = args.roughConvertOutPath
    originConvertInPath = args.originConvertInPath+"*.png"
    originConvertOutPath = args.originConvertOutPath
    trainPath = args.trainPath
    train_dir = os.path.dirname(trainPath)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    convertAllpng(roughConvertInPath, roughConvertOutPath)
    convertAllpng(originConvertInPath, originConvertOutPath)
    convertImageType(roughConvertOutPath)
    convertImageType(originConvertOutPath)
    images = []
    for f in os.listdir(roughConvertOutPath):
        if (f != '.DS_Store'):
            images.append(f)

    # make the training dataset
    for i in range(len(images)):
        uid = images[i][-8:-4]
        target1 = Image.new('1', (TARGET_WIDTH, UNIT_SIZE))
        # target2 = Image.new('1', (TARGET_WIDTH, UNIT_SIZE))
        # target3 = Image.new('1', (TARGET_WIDTH, UNIT_SIZE))
        # target4 = Image.new('1', (TARGET_WIDTH, UNIT_SIZE))
        left = 0
        right = UNIT_SIZE
        rough_image = Image.open(roughConvertOutPath+'/'+images[i])
        origin_image = Image.open(originConvertOutPath+'/'+images[i])
        imagefile = []
        imagefile.append(origin_image)
        imagefile.append(rough_image)
        for image in imagefile:
            target1.paste(image, (left, 0, right, UNIT_SIZE))
        #     target2.paste(image.transpose(Image.FLIP_LEFT_RIGHT), (left, 0, right, UNIT_SIZE))
        #     target3.paste(image.transpose(Image.FLIP_TOP_BOTTOM), (left, 0, right, UNIT_SIZE))
        #     target4.paste(image.transpose(Image.ROTATE_180), (left, 0, right, UNIT_SIZE))
            left += UNIT_SIZE
            right += UNIT_SIZE
        quality_value = 100 
        if uid in SPECIAL_UIDS:
            print("Yes")
            target1.save(trainPath+'1_'+str(i).zfill(5)+'_'+"1"+'.png', quality = quality_value)
        else:
            target1.save(trainPath+'0_'+str(i).zfill(5)+'_'+"1"+'.png', quality = quality_value)
        # target2.save(trainPath+'0_'+str(i).zfill(4)+'_'+"2"+'.png', quality = quality_value)
        # target3.save(trainPath+'0_'+str(i).zfill(4)+'_'+"3"+'.png', quality = quality_value)
        # target4.save(trainPath+'0_'+str(i).zfill(4)+'_'+"4"+'.png', quality = quality_value)