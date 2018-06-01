import numpy as np
import random
import csv
from scipy.misc import imread
from scipy.misc import imresize
from os import listdir
from os.path import isfile, join

class PrepareData:
    def __init__(self, datadir):
        """
        Keyword Arguments:
        x           -- filelist of movieclips + framename
        filelist    -- filelist min movieclips, used for label generation"""
        self.datadir = datadir
        print("Found datadir!")
        self.x, self.filelist = self.generateFileList()
        print("Filelist generated!")
        self.y = self.generateLabelList(self.getLabelsDict(), self.filelist)
        #shuffle that list for the split
        self.createSplits(self.x, self.y)
        print("Writing split finished!")

    def createSplits(self, x, y, split=[.6, .2, .2]):
        """Creates splits of the files based on the given split."""
        self.x_train = x[:int(len(x)*(split[0]))]
        self.x_val  = x[int(len(x)*(split[0]))+1:int(len(x)*(split[0]+split[1]))]
        self.x_test   = x[int(len(x)*(split[0]+split[1])):]
        self.y_train = y[:int(len(x)*(split[0]))]
        self.y_val  = y[int(len(x)*(split[0]))+1:int(len(x)*(split[0]+split[1]))]
        self.y_test   = y[int(len(x)*(split[0]+split[1])):]

        c = list(zip(self.x_train, self.y_train))
        random.shuffle(c)
        self.x_train, self.y_train = zip(*c)

        c = list(zip(self.x_test, self.y_test))
        random.shuffle(c)
        self.x_test, self.y_test = zip(*c)

        c = list(zip(self.x_val, self.y_val))
        random.shuffle(c)
        self.x_val, self.y_val = zip(*c)

        print(len(self.y_train), len(self.x_train))
        print(len(self.y_test), len(self.x_test))
        print(len(self.y_val), len(self.x_val))
        with open('/home/pieter/projects/engagement-l2tor/data/emotions/x_train3.txt', 'a') as f:
            for x in self.x_train:
                f.write(x + '\n')
        with open('/home/pieter/projects/engagement-l2tor/data/emotions/x_test3.txt', 'a') as f:
            for x in self.x_test:
                f.write(x + '\n')
        with open('/home/pieter/projects/engagement-l2tor/data/emotions/x_val3.txt', 'a') as f:
            for x in self.x_val:
                f.write(x + '\n')
        with open("/home/pieter/projects/engagement-l2tor/data/emotions/y_train3.txt", "a", newline='\n') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerows(self.y_train)
        with open("/home/pieter/projects/engagement-l2tor/data/emotions/y_test3.txt", "a", newline='\n') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerows(self.y_test)
        with open("/home/pieter/projects/engagement-l2tor/data/emotions/y_val3.txt", "a", newline='\n') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerows(self.y_val)

    def getData(self):
        img = self.loadImage(self.x[1])
        print(np.shape(img))


    def getFileNames(self, filename):
        temp = []
        with open(self.datadir + filename) as f:
            rdr = csv.reader(f)
            for row in rdr:
                temp.append(row[0][:-4])
        return temp

    def getLabelsDict(self):
        labels = {}
        for filename in ['train', 'test', 'validation']:
            with open(self.datadir + filename + ".txt") as f:
                rdr = csv.reader(f)
                for row in rdr:
                    labels[row[0][:-4]] = row[1:9]
        return labels

    def loadImage(self, filename, resize=[True, (128,128)]):
        img = imread(filename)
        if resize[0]:
            img = imresize(img, resize[1], interp='bicubic')
        return img

    def generateLabelList(self, labels, file_list_no_frame):
        label_list = []
        for k in file_list_no_frame:
            label_list.append(labels[k])
        return label_list

    def generateFileList(self):
        file_list = []
        file_list_no_frame = []
        file_dict = {}
        for group in ['train', 'test', 'validation']:
            temp = self.getFileNames(group + ".txt")
            for i in temp:
                file_dict[i] = self.getFrames(i, group)
        #now we have a dictionary containing all file names and so forth
        for key in file_dict:
            for item in file_dict[key]:
                #filelist contains complete directory path to frames
                file_list.append(key + "/" + item)
                file_list_no_frame.append(key)
        return file_list, file_list_no_frame

    def getFrames(self, filename, folder):
        """gets filenames of frames"""
        pth = self.datadir + folder + '/' + filename + '/'
        filelist = []
        for f in listdir(pth):
            if isfile(join(pth, f)):
                filelist.append(f)
        return filelist

def main():
    data_dir = "/home/pieter/data/emoreact/"
    prep = PrepareData(data_dir)

if __name__=="__main__":
    main()
