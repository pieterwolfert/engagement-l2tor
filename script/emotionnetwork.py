import numpy as np
import csv
from preprocessing import Preprocessing

data_dir = "/home/pieter/data/emoreact/"

def main():
    prep = Preprocessing(data_dir)
    training_filenames = prep.getFileNames("train.txt")
    training_labels = prep.getLabels("train.txt", len(training_filenames))
    #todo resize image to 128x128 on the fly, pick one image per movieclip, train using the labels

if __name__=="__main__":
    main()
