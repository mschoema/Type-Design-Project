from PIL import Image
import os.path
import glob
import os
import string
import pickle

def pickle_examples(paths, out_path):
    print(len(paths))
    with open(out_path, 'wb') as ft:
        for p in paths:
            label = 0
            with open(p, 'rb') as f:
                img_bytes = f.read()
                example = (label, img_bytes)
                pickle.dump(example, ft)

def convertAllpng(convertInPath, convertOutPath):
    for pngfile in glob.glob(convertInPath):
        convertpng(pngfile, convertOutPath)

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

def main():
    roughConvertInPath = "../outputFiles/Soft_song/rough_1000/*.png"
    roughConvertOutPath = "../outputFiles/Soft_song/rough_256/"
    convertAllpng(roughConvertInPath, roughConvertOutPath)

    out_path = "../outputFiles/Soft_song/images.obj"
    pickle_examples(sorted(glob.glob(os.path.join(roughConvertOutPath, "*.png"))), out_path=out_path)

if __name__ == '__main__':
    main()