from keras.models import model_from_json
from preprocessing import Preprocessing
from keras import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def precisionsMultilabel(pred, y_test):
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
    print("Number of total equal rows: {}".format(sum_pred / len(y_test)))
    labels = ["Curiosity", "Uncertainty", "Excitement", "Happiness",\
        "Surprise", "Disgust", "Fear", "Frustration"]
    table = []
    for i, item in enumerate(labels):
        precision = precision_score(column(y_test, i), column(y_test_pred, i))
        recall = recall_score(column(y_test, i), column(y_test_pred, i))
        f1 = f1_score(column(y_test, i), column(y_test_pred, i))
        table.append([item, precision, recall, f1])
    print(tabulate(table, headers=(["Label", "Precision", "Recall", "F1"])))

def column(matrix, i):
    return [int(row[i]) for row in matrix]

def runInference():
    #model 1
    x_test, y_test = prep.getTestData(img_shape=(96,96))
    with open('./models/model_json9696.txt', 'r') as f:
        ymlmodel = f.read()
    model = model_from_json(ymlmodel)
    model.load_weights('./models/emotions9696_200_256.hdf5')
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    pred = model.predict(x_test)
    print("Results of 96x96 pixels classifier: \n")
    precisionsMultilabel(pred, y_test)
    #model 2
    x_test, y_test = prep.getTestData(img_shape=(128,128))
    with open('./models/model_json.txt', 'r') as f:
        ymlmodel = f.read()
    model = model_from_json(ymlmodel)
    model.load_weights('./models/emotions.hdf5')
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    pred2 = model.predict(x_test)
    print("Results of 128x128 pixels classifier: \n")
    precisionsMultilabel(pred2, y_test)
    combined_predictions = (np.add(np.array(pred), np.array(pred2))) / 2
    print("Results of the two combined: \n")
    precisionsMultilabel(combined_predictions, y_test)
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    #precisionsMultilabel(pred, y_test)

def main():
    datadir = "/home/pieter/projects/engagement-l2tor/data/emotions/"
    prep = Preprocessing(datadir, "x_train.txt", "x_test.txt",\
        "y_train.txt", "y_test.txt")
    #runInference(prep)
    x_train, y_train = prep.getTrainData(True, (96,96))
    tot = len(y_train)
    for i in range(8):
        print(np.sum(y_train[:,i].astype(np.float)))


if __name__ == '__main__':
    main()
