import os
import layouts
from class_definitions import BoundingBox
from PIL import Image

def main():
    in_dir = "../characterImages/font/"
    img_list = os.listdir(in_dir)
    n = len(img_list)
    print(n)
    fonts = [img_list[x][:-9] for x in range(n) if x%3 == 0]
    chars = [img_list[x] for x in range(n) if x%3 == 0]
    comps1 = [img_list[x] for x in range(n) if x%3 == 1]
    comps2 = [img_list[x] for x in range(n) if x%3 == 2]
    box1 = BoundingBox(44, 39, 848, 529)
    box2 = BoundingBox(197, 546, 609, 408)
    boxes = [box1, box2]
    l = int(n/3)
    print(l)

    for i in range(l):
        font = fonts[i]
        char_dir = os.path.join(in_dir, chars[i])
        comp1_dir = os.path.join(in_dir, comps1[i])
        comp2_dir = os.path.join(in_dir, comps2[i])
        char_img = Image.open(char_dir)
        comp1_img = Image.open(comp1_dir)
        comp2_img = Image.open(comp2_dir)
        comps = [comp1_img, comp2_img]
        rough_img = layouts.applyLayout(2, comps)
        precise_img = layouts.applyPreciseLayout(boxes, comps)
        target1 = Image.new('1', (2000, 1000))
        target2 = Image.new('1', (2000, 1000))
        target1.paste(char_img, (0, 0, 1000, 1000))
        target1.paste(rough_img, (1000, 0, 2000, 1000))
        target2.paste(char_img, (0, 0, 1000, 1000))
        target2.paste(precise_img, (1000, 0, 2000, 1000))

        target1.save("../outputFiles/667a/rough/{}.png".format(font))
        target2.save("../outputFiles/667a/precise/{}.png".format(font))


if __name__ == '__main__':
    main()