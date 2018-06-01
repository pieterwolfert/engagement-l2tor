import numpy as np
import csv
from keras.utils import plot_model
from preprocessing import Preprocessing
from keras.layers import Activation, Input, Conv2D, Dense, Dropout, \
    MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D
from keras import layers
from keras.models import Model
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization

def getmodel(image_shape):
    #part 1
    img_input = Input(shape=image_shape)
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
    output = Dense(1, activation='softmax', name='predictions')(x)
    model = Model(inputs=img_input, outputs=output)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def train(x_train, y_train, x_val, y_val, image_shape):
    nr_epochs = 200
    batch = 512
    model = getmodel(image_shape)
    lr_plateau = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5)
    checkpoint = ModelCheckpoint(filepath='./models/' + 'happiness128128v1' + str(nr_epochs) +\
                            '_' + str(batch) + '.hdf5',\
                             verbose=1, save_best_only=True)
    #plot_model(model, to_file="architecture.png")
    model.fit(x_train, y_train, epochs=nr_epochs, batch_size=batch,\
        callbacks=[lr_plateau, checkpoint],\
         validation_data=(x_val, y_val))

def modelStructure(image_shape):
    model = getmodel(image_shape)
    json_ml = model.to_json()
    with open('./models/model_structure_json.txt', 'w') as f:
        f.write(json_ml)

def main():
    image_shape = (128, 128, 3)
    datadir = "/home/awolfert/projects/engagement-l2tor/data/emotions/"
    prep = Preprocessing(datadir, "x_train3.txt", "x_test3.txt", "x_val3.txt",\
        "y_train3.txt", "y_test3.txt", "y_val3.txt")
    x_train, y_train, x_val, y_val = prep.getTrainData(trim=True, img_shape=image_shape)
    train(x_train, y_train[:,3], x_val, y_val[:,3], image_shape)
    #modelStructure(image_shape)

if __name__=="__main__":
    main()
