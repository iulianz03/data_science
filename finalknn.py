import numpy
from numpy import genfromtxt
import os
from sklearn.neighbors import KNeighborsClassifier
import csv


os.chdir('/home/dt/Documents/School/2016-2017/Spring/Machine learning/Assignment 2/Data')

X_train_all = numpy.load("X_train_all.npy")
X_train_sdfl = numpy.load("X_train_sdfl.npy")
X_train_hsd = numpy.load("X_train_hsd.npy")
X_train = numpy.concatenate((X_train_all, X_train_sdfl), axis = 1)

X_test_all  = numpy.load("X_test_all.npy")
X_test_sdfl = numpy.load("X_test_sdfl.npy")
X_test = numpy.concatenate((X_test_all, X_test_sdfl), axis = 1)
y_train = genfromtxt('meg-train-label.csv', delimiter=',', skip_header = 1, usecols = 1)


#Zscale the items
from sklearn.preprocessing import StandardScaler
zscore = StandardScaler()
X_train_z  = zscore.fit_transform(X_train)
X_test_z   = zscore.transform(X_test)


#fit knn model on training data and predict on test data
model = KNeighborsClassifier(weights ='distance', algorithm = 'auto', n_neighbors = 4)
model.fit(X_train_z, y_train)
y_pred = model.predict(X_test_z)


#enumarate and write to csv file

num = 0
with open('result.csv','w') as f1:
    headerwriter = csv.DictWriter(f1, fieldnames = ["RowIndex", "ClassID"], delimiter = ',')
    headerwriter.writeheader()
    writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
    for prediction in y_pred:
        row = (num, int(prediction))
        writer.writerow(row)
        num += 1



from sklearn.model_selection import cross_val_score
print(cross_val_score(model, X_train_z, y_train, cv=10))
