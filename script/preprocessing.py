import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imresize

class Preprocessing():
    def __init__(self, datadir,  x_train, x_test, y_train, y_test):
        self.datadir = datadir
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def getTrainData(self, trim=True):
        """Gets training data, normalizes data, ready for training."""
        trim_size = 1000
        img_size = (128,128)
        y_train = []
        with open(self.datadir + self.y_train) as f:
            rdr = csv.reader(f)
            for row in rdr:
                y_train.append(row)
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
            imgs[i] = self.loadImage(flnm)
        imgs = imgs/255
        imgs -= np.mean(imgs, axis=0)
        imgs = imgs/np.std(imgs, axis=0)
        return imgs, y_train


    def getTestData(self, trim_size=1000):
        img_size = (128,128)
        y_test = []
        with open(self.datadir + self.y_test) as f:
            rdr = csv.reader(f)
            for row in rdr:
                y_test.append(row)
        y_test = np.asarray(y_test)
        with open(self.datadir + self.x_test) as f:
            x_test = f.readlines()
            x_test = [x.strip() for x in x_test]
        y_test = y_test[:trim_size]
        x_test = x_test[:trim_size]
        imgs = np.zeros((len(x_test), img_size[0], img_size[0], 3))
        for i, x in enumerate(x_test):
            flnm = self.datadir + x
            imgs[i] = self.loadImage(flnm)
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
    prep = Preprocessing(datadir, "x_train.txt", "x_test.txt",\
        "y_train.txt", "y_test.txt")
    prep.getTrainData()


if __name__=="__main__":
    main()
