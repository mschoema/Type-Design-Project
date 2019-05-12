from matplotlib import pyplot as plt

def display_array(data):
    if len(data.shape) == 4:
        data = data[0,:,:,0]
    im = plt.imshow(data, interpolation='nearest')
    plt.colorbar(im)
    plt.show()