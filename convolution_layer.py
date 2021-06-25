import matplotlib
import numpy as np
import sklearn.tree
import sklearn.tree as sk
import pandas
import csv_dataset

def convolute():

    # dataset = csv_dataset.save_dataset()
    dataset = pandas.read_csv("dataset.csv")


    xtrain = dataset[0:21000, 1:]
    train_label = dataset[0:21000, 0]

    clf = sklearn.tree.DecisionTreeClassifier()
    clf.fit(xtrain, train_label)

    xtest = dataset[21000:, 1:]
    actual_label = dataset[21000: ,0]



    print(xtest)
    print(xtrain)
    # matplotlib.pyplot.imshow(255-d, cmap = 'gray')
    # print(clf.predict([xtest[0]]))
    # matplotlib.pyplot.show()


convolute()
print("here")