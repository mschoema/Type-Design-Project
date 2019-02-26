from PIL import Image, ImageChops

IMAGES_WIDTH = 1000
IMAGES_HEIGTH = 1000
IMAGES_SIZE = (IMAGES_WIDTH, IMAGES_HEIGTH)

layouts = {
    0: [(0,0,1,1)],
    1: [(0,0,0.5,1), (0.5,0,0.5,1)],
    2: [(0,0,1,0.5), (0,0.5,1,0.5)],
    3: [(0,0,1,1), (0.5,0.5,0.5,0.5)],
    4: [(0,0,1,1), (0.5,0,0.5,0.5)],
    5: [(0,0,1,1), (0,0.5,0.5,0.5)],
    6: [(0,0,1,1), (0.25,0.5,0.5,0.5)],
    7: [(0,0,1,1), (0.25,0,0.5,0.5)],
    8: [(0,0,1,1), (0.5,0.25,0.5,0.5)],
    9: [(0,0,1,1), (0.25,0.25,0.5,0.5)],
    10: [(0,0,0.33,1), (0.33,0,0.34,1), (0.67,0,0.33,1)],
    11: [(0,0,1,0.33), (0,0.33,1,0.34), (0,0.67,1,0.33)],
    12: [(0,0,1,0.5), (0,0.5,0.5,0.5), (0.5,0.5,0.5,0.5)],
    13: [(0,0,0.5,0.5), (0.5,0,0.5,0.5), (0,0.5,1,0.5)],
    14: [(0,0,1,1), (0,0.25,0.5,0.5), (0.5,0.25,0.5,0.5)],
    15: [(0,0,0.5,0.5), (0.5,0,0.5,0.5), (0,0.5,0.5,0.5), (0.5,0.5,0.5,0.5)],
    16: [(0,0,1,0.33), (0,0.33,0.5,0.34), (0.5,0.33,0.5,0.34), (0,0.67,1,0.33)],
    17: [(0,0,1,0.25), (0,0.25,1,0.25), (0,0.5,0.33,0.5), (0.33,0.5,0.34,0.5), (0.67,0.5,0.33,0.5)],
    18: [(0,0,0.5,0.33), (0.5,0,0.5,0.33), (0,0.33,1,0.34), (0,0.67,0.5,0.33), (0.5,0.67,0.5,0.33)],
    19: [(0,0,1,0.33), (0,0.33,0.33,0.34), (0.33,0.33,0.34,0.34), (0.67,0.33,0.33,0.34), (0,0.67,1,0.33)],
    20: [(0,0,1,1), (0.33,0,0.34,1)],
    21: [(0,0,0.5,0.5), (0,0.5,0.5,0.5), (0.5,0,0.5,1)],
    22: [(0,0,0.5,1), (0.5,0,0.5,0.5), (0.5,0.5,0.5,0.5)],
    23: [(0,0,1,1), (0.33,0.33,0.67,0.34), (0.33,0.67,0.67,0.33)],
    24: [(0,0,1,1), (0.33,0.33,0.34,0.67), (0.67,0.33,0.33,0.67)],
    25: [(0,0,1,1), (0,0,0.5,0.5), (0.5,0,0.5,0.5)],
    26: [(0,0,1,0.25), (0,0.25,1,0.25), (0,0.5,1,0.25), (0,0.75,1,0.25)],
    27: [(0,0,1,1), (0,0,0.5,0.5), (0.5,0,0.5,0.5), (0,0.5,0.5,0.5), (0.5,0.5,0.5,0.5)]
}

# def applyLayout(layout, image):
#     print(layout)
#     outImage = Image.new("1", IMAGES_SIZE, 1)
#     comps = layouts.get(layout, layouts.get(0))
#     for comp in comps:
#         x, y, dx, dy = int(comp[0]*IMAGES_WIDTH), int(comp[1]*IMAGES_HEIGTH), int(comp[2]*IMAGES_WIDTH), int(comp[3]*IMAGES_HEIGTH)
#         tempImage = image.resize((dx,dy))
#         print(tempImage.size)
#         outImage.paste(tempImage, (x, y, x + dx, y + dy), ImageChops.invert(tempImage))
#     return outImage

def applyLayout(layout, images):
    comps = layouts.get(layout, layouts.get(0))
    if (len(images) != len(comps)):
        print("incorrect lengths")
        # Throw error
    else:
        outImage = Image.new("1", IMAGES_SIZE, 1)
        for i in range(len(images)):
            comp = comps[i]
            image = images[i]
            x, y, dx, dy = int(comp[0]*IMAGES_WIDTH), int(comp[1]*IMAGES_HEIGTH), int(comp[2]*IMAGES_WIDTH), int(comp[3]*IMAGES_HEIGTH)
            tempImage = image.resize((dx,dy))
            outImage.paste(tempImage, (x, y, x + dx, y + dy), ImageChops.invert(tempImage))
    
    return outImage