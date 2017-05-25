import numpy
import os
import csv
import scipy

os.chdir('/home/dt/Documents/School/2016-2017/Spring/Machine learning/Assignment 2/Data')

X_train = numpy.load("meg-train-features.npy")
X_test  = numpy.load("meg-test-features.npy")

#feature variables train and test
X_train_sdfl = []
X_test_sdfl  = []

#function to get sd of fitted line
def fitter(y):
    y.tolist
    x = list(range(200))
    from scipy.stats import linregress
    return(linregress(x, y)[4])


for id in X_train:
    idtrain = id.reshape(204,200) #reshapes the dataset into a workable array
     #get sd of plottedline of the channels
    fidtrain = numpy.apply_along_axis(fitter, axis=1, arr=idtrain)
    fidtrain.tolist
    X_train_sdfl.append(fidtrain)

for id in X_test:
    idtest = id.reshape(204,200) #reshapes the dataset into a workable array
     #get sd of plottedline of the channels
    fidtrain = numpy.apply_along_axis(fitter, axis=1, arr=idtest)
    fidtrain.tolist
    X_test_sdfl.append(fidtrain)

numpy.save('X_train_sdfl'   , X_train_sdfl )
numpy.save('X_test_sdfl'    , X_test_sdfl  )
    
