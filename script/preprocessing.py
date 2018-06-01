import numpy as np
import csv
from scipy.misc import imread
from scipy.misc import imresize

class Preprocessing():
    def __init__(self, datadir,  x_train, x_test, x_val,\
        y_train, y_test, y_val):
        self.datadir = datadir
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val

    def getTrainData(self, trim, img_shape):
        """Gets training data, normalizes data, ready for training."""
        trim_size = 1000
        img_size = (img_shape[0],img_shape[1])
        y_train = self.getLabels(self.y_train)
        if trim:
            y_train = y_train[:trim_size]
        y_train = np.asarray(y_train)
        with open(self.datadir + self.x_train) as f:
            x_train = f.readlines()
            x_train = [x.strip() for x in x_train]
        if trim:
            x_train = x_train[:trim_size]
        imgs = np.zeros((len(x_train), img_size[0], img_size[0], 3))
        for i, x in enumerate(x_train):
            flnm = self.datadir + x
            imgs[i] = self.loadImage(flnm, resize=[True, img_size])
        imgs = imgs/255
        imgs -= np.mean(imgs, axis=0)
        imgs = imgs/np.std(imgs, axis=0)

        y_val = self.getLabels(self.y_val)
        if trim:
            y_val = y_val[:trim_size]
        y_train = np.asarray(y_train)
        with open(self.datadir + self.x_val) as f:
            x_val = f.readlines()
            x_val = [x.strip() for x in x_train]
        if trim:
            x_val = x_val[:trim_size]
        imgs_val = np.zeros((len(x_val), img_size[0], img_size[0], 3))
        for i, x in enumerate(x_val):
            flnm = self.datadir + x
            imgs_val[i] = self.loadImage(flnm, resize=[True, img_size])
        imgs_val = imgs_val/255
        imgs_val -= np.mean(imgs_val, axis=0)
        imgs_val = imgs_val/np.std(imgs_val, axis=0)

        return imgs, y_train, imgs_val, y_val

    def getLabels(self, labelFile):
        temp = []
        with open(self.datadir + labelFile) as f:
            rdr = csv.reader(f)
            for row in rdr:
                temp.append(row)
        return np.asarray(temp)

    def getTestData(self, img_shape):
        trim_size = 2000
        img_size = (img_shape[0],img_shape[1])
        y_test = self.getLabels(self.y_test)
        with open(self.datadir + self.x_test) as f:
            x_test = f.readlines()
            x_test = [x.strip() for x in x_test]
        y_test = y_test[:trim_size]
        x_test = x_test[:trim_size]
        imgs = np.zeros((len(x_test), img_size[0], img_size[0], 3))
        for i, x in enumerate(x_test):
            flnm = self.datadir + x
            imgs[i] = self.loadImage(flnm, resize=[True, img_size])
        imgs = imgs/255
        imgs -= np.mean(imgs, axis=0)
        imgs = imgs/np.std(imgs, axis=0)
        return imgs, y_test


    def loadImage(self, filename, resize=[True, (128,128)]):
        img = imread(filename)
        if resize[0]:
            img = imresize(img, resize[1], interp='bicubic', mode="RGB")
        return img

def main():
    datadir = "/home/pieter/projects/engagement-l2tor/data/emotions/"
    prep = Preprocessing(datadir, "x_train3.txt", "x_test3.txt", "x_val3.txt",\
        "y_train3.txt", "y_test3.txt", "y_val3.txt")
    xtrain, ytrain, xval, yval = prep.getTrainData(True, (128,128))

if __name__=="__main__":
    main()
