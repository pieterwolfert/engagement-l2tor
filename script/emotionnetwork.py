import numpy as np
import csv
from preprocessing import Preprocessing
import keras

data_dir = "/home/pieter/data/emoreact/"

def model():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def main():
    prep = Preprocessing(data_dir)
    x_train, y_train, x_test, y_test = prep.loadData("train.txt", "test.txt")
    model = model()
    model.fit(data, labels, epochs=20, batch_size=32)
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)

if __name__=="__main__":
    main()
