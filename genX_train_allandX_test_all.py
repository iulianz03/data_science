import numpy
import os
import csv
import scipy
from scipy.stats import skew
from scipy.stats import kurtosis

os.chdir('/home/dt/Documents/School/2016-2017/Spring/Machine learning/Assignment 2/Data')

X_train = numpy.load("meg-train-features.npy")
X_test  = numpy.load("meg-test-features.npy")

#function to get detrended variance of the MEG channels
def detrender(x):
    import scipy.signal
    return(scipy.signal.detrend(x))


#feature variables train and test
X_train_all = []
X_test_all  = []

for id in X_train:
    idtrain = id.reshape(204,200) #reshapes the dataset into a workable array
    #detrends the channels
    didtrain = numpy.apply_along_axis(detrender, axis=1, arr=idtrain )

    #get non-detrended features mean and standard deviation
    hsd = numpy.std(idtrain, axis = 1)
    hvar = numpy.var(idtrain, axis = 1)
    hm  = numpy.mean(idtrain, axis = 1)
    ha = numpy.average(idtrain, axis = 1)

    #get the detrended features statistics
    dm = numpy.mean(didtrain, axis = 1)
    da = numpy.average(didtrain, axis = 1)
    dsd = numpy.std(didtrain, axis = 1)
    dvar = numpy.var(didtrain, axis = 1)
    dskew = skew(didtrain, axis = 1)
    dkurt = kurtosis(didtrain, axis = 1)

    #featurize
    all = numpy.concatenate((hm,ha,hsd,hvar,dm, da, dsd, dvar,dskew,dkurt),axis = 0)
    all.tolist
    X_train_all.append(all)

    
for id in X_test:

    idtest = id.reshape(204,200) #reshapes the dataset into a workable array

    #detrends the channels
    didtest = numpy.apply_along_axis(detrender, axis=1, arr=idtest )

    #get non-detrended features mean and standard deviation
    hsd = numpy.std(idtest, axis = 1)
    hvar = numpy.var(idtest, axis = 1)
    hm  = numpy.mean(idtest, axis = 1)
    ha = numpy.average(idtest, axis = 1)

    #get the detrended statistics.
    dm = numpy.mean(didtest, axis = 1)
    da = numpy.average(didtest, axis = 1)
    dsd = numpy.std(didtest, axis = 1)
    dvar = numpy.var(didtest, axis = 1)
    dskew = skew(didtest, axis = 1)
    dkurt = kurtosis(didtest, axis = 1)

    #featurize
    all = numpy.concatenate((hm,ha,hsd,hvar,dm, da, dsd, dvar,dskew,dkurt),axis = 0)
    all.tolist
    X_test_all.append(all)

#save the data
numpy.save('X_train_all'   , X_train_all )
numpy.save('X_test_all'    , X_test_all  )

