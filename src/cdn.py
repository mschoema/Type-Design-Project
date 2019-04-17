### Character Decomposition Network ###

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Dropout, LeakyReLU, Flatten, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils import plot_model
import csv
from layouts import layouts

def NN_model(n_filters=16, dropout=0.5):
    Input_character = Input(shape=(1000,1000,1), name='character')
    Input_component1 = Input(shape=(1000,1000,1), name='component1')
    Input_component2 = Input(shape=(1000,1000,1), name='component2')

    # A convolution block consisting of two layers of convolution + batchnormalization + activation (leaky relu)
    def conv_block(input_tensor, n_filters, kernel_size=3):
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="random_uniform",
                   padding="same")(input_tensor)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="random_uniform",
                   padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    x = conv_block(Input_character, n_filters=n_filters, kernel_size=5)
    x = MaxPooling2D((2,2)) (x) # -> (500,500)
    x = Dropout(dropout*0.5)(x)

    x = conv_block(x, n_filters=n_filters*2)
    x = MaxPooling2D((2,2)) (x) # -> (250,250)
    x = Dropout(dropout)(x)

    x = conv_block(x, n_filters=n_filters*4)
    x = MaxPooling2D((2,2)) (x) # -> (125,125)
    x = Dropout(dropout)(x)

    x = conv_block(x, n_filters=n_filters*8)
    x = MaxPooling2D((2,2)) (x) # -> (64,64)
    x = Dropout(dropout)(x)

    c1 = conv_block(Input_component1, n_filters=n_filters, kernel_size=5)
    c1 = MaxPooling2D((2,2)) (c1) # -> (500,500)
    c1 = Dropout(dropout*0.5)(c1)

    c1 = conv_block(c1, n_filters=n_filters*2)
    c1 = MaxPooling2D((2,2)) (c1) # -> (250,250)
    c1 = Dropout(dropout)(c1)

    c1 = conv_block(c1, n_filters=n_filters*4)
    c1 = MaxPooling2D((2,2)) (c1) # -> (125,125)
    c1 = Dropout(dropout)(c1)

    c1 = conv_block(c1, n_filters=n_filters*8)
    c1 = MaxPooling2D((2,2)) (c1) # -> (64,64)
    c1 = Dropout(dropout)(c1)

    c2 = conv_block(Input_component2, n_filters=n_filters, kernel_size=5)
    c2 = MaxPooling2D((2,2)) (c2) # -> (500,500)
    c2 = Dropout(dropout*0.5)(c2)

    c2 = conv_block(c2, n_filters=n_filters*2)
    c2 = MaxPooling2D((2,2)) (c2) # -> (250,250)
    c2 = Dropout(dropout)(c2)

    c2 = conv_block(c2, n_filters=n_filters*4)
    c2 = MaxPooling2D((2,2)) (c2) # -> (125,125)
    c2 = Dropout(dropout)(c2)

    c2 = conv_block(c2, n_filters=n_filters*8)
    c2 = MaxPooling2D((2,2)) (c2) # -> (64,64)
    c2 = Dropout(dropout)(c2)

    y1 = concatenate([x,c1], axis=3)
    y2 = concatenate([x,c2], axis=3)

    y1 = conv_block(y1, n_filters=n_filters*16)
    y2 = conv_block(y2, n_filters=n_filters*16)
    y1 = MaxPooling2D((2,2)) (y1) # -> (32,32)
    y2 = MaxPooling2D((2,2)) (y2)
    y1 = Dropout(dropout)(y1)
    y2 = Dropout(dropout)(y2)

    y1 = conv_block(y1, n_filters=n_filters*32)
    y2 = conv_block(y2, n_filters=n_filters*32)
    y1 = MaxPooling2D((2,2)) (y1) # -> (16,16)
    y2 = MaxPooling2D((2,2)) (y2)
    y1 = Dropout(dropout)(y1)
    y2 = Dropout(dropout)(y2)

    y1 = Flatten() (y1)
    y2 = Flatten() (y2)

    y1 = Dense(128, activation='relu') (y1)
    y2 = Dense(128, activation='relu') (y2)

    output1 = Dense(4, activation='sigmoid') (y1)
    output2 = Dense(4, activation='sigmoid') (y2)

    output = concatenate([output1, output2])

    model = Model(inputs=[Input_character, Input_component1, Input_component2], outputs=[output])
    return model

def main():
    model = CDN_model()
    plot_model(model, to_file='../testFiles/model.png')
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    model.summary()

if __name__ == "__main__":
    main()