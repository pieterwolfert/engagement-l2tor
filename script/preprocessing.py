import numpy as np
import csv
from scipy.misc import imread
from scipy.misc import imresize

class Preprocessing:
    def __init__(self, datadir):
        self.datadir = datadir

    def loadData(self, train, test):
        x_train_filenames = self.getFileNames(train)
        x_test_filenames = self.getFileNames(test)
        y_train = self.getLabels(train, len(x_train_filenames))
        y_test = self.getLabels(test, len(x_test_filenames))
        self.x_train, self.y_train = self.loadImages(x_train_filenames,\
         y_train, "train/", (128,128))
        self.x_test, self.y_test = self.loadImages(x_test_filenames,\
            y_test, "test/", (128,128))
        return self.x_train, self.y_train, self.x_test, self.y_test

    def getFileNames(self, filename):
        temp = []
        with open(self.datadir + filename) as f:
            rdr = csv.reader(f)
            for row in rdr:
                temp.append(row[0][:-4])
        return temp

    def getLabels(self, filename, length):
        labels = np.zeros(shape=(length, 8))
        with open(self.datadir + filename) as f:
            rdr = csv.reader(f)
            for i, item in enumerate(rdr):
                labels[i] = item[1:9]
        return labels

    def loadImages(self, filenames, labels, folder, size):
        ext = ["0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png",\
        "7.png", "8.png", "9.png", "10.png"]
        #ext = ["0.png"]
        x_train = []
        y_train = []
        for x, y in zip(filenames, labels):
            for e in ext:
                try:
                    tmp = imread(self.datadir + folder + x + "/" + e)
                    x_train.append(imresize(tmp, size, interp='bicubic'))
                    y_train.append(y)
                except FileNotFoundError as f:
                    pass
        return np.asarray(x_train), np.asarray(y_train)
