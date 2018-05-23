import numpy as np
import csv

class Preprocessing:
    def __init__(self, datadir):
        self.datadir = datadir
        self.filenames = []

    def getFileNames(self, filename):
        with open(self.datadir + filename) as f:
            rdr = csv.reader(f)
            for row in rdr:
                self.filenames.append(row[0][:-4])
        return self.filenames

    def getLabels(self, filename, length):
        self.labels = np.zeros(shape=(length, 8))
        with open(self.datadir + filename) as f:
            rdr = csv.reader(f)
            for i, item in enumerate(rdr):
                self.labels[i] = item[1:9]
        return self.labels
