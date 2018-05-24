import numpy as np
import csv
from keras.utils import plot_model
from preprocessing import Preprocessing
from keras.layers import Activation, Input, Conv2D, Dense, Dropout, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

data_dir = "/home/pieter/data/emoreact/"

def getmodel():
    #part 1
    img_input = Input(shape=(128,128,3))
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=(2, 2), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #this part gets repeated 4 times
    residual = Conv2D(128, (1, 1), strides=(2, 2),\
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    #2
    residual = Conv2D(128, (1, 1), strides=(2, 2),\
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    #3
    residual = Conv2D(128, (1, 1), strides=(2, 2),\
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    #4
    residual = Conv2D(128, (1, 1), strides=(2, 2),\
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(64, (2, 2), strides=(2, 2), use_bias=False)(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(8, activation='softmax', name='predictions')(x)
    model = Model(inputs=img_input, outputs=output)
    return model

def main():
    prep = Preprocessing(data_dir)
    x_train, y_train, x_test, y_test = prep.loadData("train.txt", "test.txt")
    print(np.shape(x_train))
    model = getmodel()
    plot_model(model, to_file="architecture.png")
    #model.fit(x_train, y_train, epochs=100, batch_size=64)
    #score = model.evaluate(x_test, y_test, batch_size=32)


if __name__=="__main__":
    main()
