from PIL import Image
import os.path
import glob
import os
import string
import argparse

UNIT_SIZE = 256
TARGET_WIDTH = 2 * UNIT_SIZE

# convert the size of png to 256*256
def convertpng(pngfile,outdir,width=256,height=256):
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
	parser.add_argument('--roughConvertInPath', type=str, default = None)
	parser.add_argument('--roughConvertOutPath', type=str, default = None)
	parser.add_argument('--originConvertInPath', type=str, default = None)
	parser.add_argument('--originConvertOutPath', type=str, default = None)
	parser.add_argument('--trainPath', type=str, default = None)

	args = parser.parse_args()
	roughConvertInPath = args.roughConvertInPath+"*.png"
    	roughConvertOutPath = args.roughConvertOutPath
    	originConvertInPath = args.originConvertInPath+"*.png"
    	originConvertOutPath = args.originConvertOutPath
    	trainPath = args.trainPath

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
	    target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE))
	    left = 0
	    right = UNIT_SIZE
	    rough_image = Image.open(roughConvertOutPath+'/'+images[i])
	    origin_image = Image.open(originConvertOutPath+'/'+images[i])
	    imagefile = []
	    imagefile.append(origin_image)
	    imagefile.append(rough_image)
	    for image in imagefile:
	        target.paste(image, (left, 0, right, UNIT_SIZE))
	        left += UNIT_SIZE 
	        right += UNIT_SIZE 
	        quality_value = 100 
	        target.save(trainPath+'0_'+str(i).zfill(4)+'.jpg', quality = quality_value)


