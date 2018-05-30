from keras.models import model_from_json
from preprocessing import Preprocessing
from keras import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

def precisionsMultilabel():
    pr = []
    for p in pred:
        pr.append([1 if x > 0.5 else 0 for x in p])
    y_test_pred = pr
    sum_pred = 0
    for x, y in zip(y_test, y_test_pred):
        if int(x[0]) == y[0] and int(x[1]) == y[1] and int(x[2]) == y[2]\
            and int(x[3]) == y[3] and int(x[4]) == y[4] and int(x[5]) == y[5]\
            and int(x[6]) == y[6] and int(x[7]) == y[7]:
            sum_pred += 1
    print(sum_pred / len(y_test))
    happiness = 0
    for x, y in zip(y_test, y_test_pred):
        if(int(x[3]) == y[3]) and y[3] == 1:
            happiness += 1
    ones = 0
    for x in y_test:
        if int(x[3]) == 1:
            ones += 1

    print(happiness, ones)
    precision = precision_score(column(y_test, 3), column(y_test_pred, 3))
    print(precision)
    recall = recall_score(column(y_test, 3), column(y_test_pred, 3))
    print(recall)
    f1 = f1_score(column(y_test, 3), column(y_test_pred, 3))
    print(f1)

def column(matrix, i):
    return [int(row[i]) for row in matrix]

def main():
    classes = ["Curiosity", "Uncertainty", "Excitement", "Happiness",\
        "Surprise", "Disgust", "Fear", "Frustration"]
    datadir = "/home/pieter/projects/engagement-l2tor/data/emotions/"
    prep = Preprocessing(datadir, "x_train.txt", "x_test.txt",\
        "y_train.txt", "y_test.txt")
    x_test, y_test = prep.getTestData()
    with open('./models/model_json.txt', 'r') as f:
        ymlmodel = f.read()
    model = model_from_json(ymlmodel)
    model.load_weights('./models/emotions_1.hdf5')
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    #score = model.evaluate(x_test, y_test, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    pred = model.predict(x_test)

if __name__ == '__main__':
    main()
