import os
import sys
from PIL import Image

letter = {0:'a', 1:'b', 2:'c'}

def main(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    images_names = os.listdir(in_dir)
    for image_name in images_names:
        image = Image.open(os.path.join(in_dir,image_name))
        for i in range(16):
            for j in range(3):
                newName = image_name[:-4] + "_" + str(i+1) + "_" + letter.get(j) + ".png"
                img_part = image.crop((256*j,256*i,256*(j+1),256*(i+1)))
                img_part.load()
                img_part.save(os.path.join(out_dir, newName))

if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    main(in_dir, out_dir)