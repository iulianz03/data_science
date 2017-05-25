#extract train features

import csv
import codecs
import numpy

file_name = 'airline-train-features.csv'
X_train = []
error_x = []

counter = 0
with codecs.open(file_name, "r",encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    try:
        for row in reader:
            X_train.append(row)
            counter += 1
    except UnicodeDecodeError:
        print(counter)
        error.append(row[0])
        counter += 1
        pass

len_X_train = len(X_train)
#delete header
header_feat = X_train[0]
del(X_train[0])

#extract train labels
file_name = 'airline-train-label.csv'
y_train = []
error_y = []


counter = 0
with codecs.open(file_name, "r",encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    try:
        for row in reader:
            y_train.append(row[1])
            counter += 1
    except UnicodeDecodeError:
        error_y.append(row[0])
        counter += 1
        pass

#delete header
header_feat = y_train[0]
y_train = y_train[1:]
if not len(X_train) == len(y_train):
    print("error, lenghts of y_train and X_train do not match")

y_train = numpy.asarray(y_train)
 

#seperate the columns
ids           = []
airline       = []
username      = []
retweet_count = []
text          = []
timestamp     = []
location      = []

for row in X_train:
    ids.append(          int(row[0]))
    airline.append(          row[1])
    username.append(         row[2])
    retweet_count.append(int(row[3]))
    text.append(             row[4])
    timestamp.append(        row[5])
    location.append(         row[6])
    
#check if everything is ok.
if file_name == 'airline-train-features.csv':
    if not len(location) == 11640:
        print("something went wrong: number of instances in data:",len(location),"does not match original number of instances:", len_X_train - 1)


#loading the test data
file_name_test = 'airline-test-features.csv'
X_test = []
counter = 0
with codecs.open(file_name_test, "r",encoding='utf-8', errors='ignore') as f:
    t_reader = csv.reader(f)
    try:
        for row in t_reader:
            X_test.append(row)
            counter += 1
    except UnicodeDecodeError:
        print(counter)
        error.append(row[0])
        counter += 1
        pass
header_feat2 = X_test[0]
del(X_test[0])

len_X_test = len(X_test)
tids           = []
tairline       = []
tusername      = []
tretweet_count = []
ttext          = []
ttimestamp     = []
tlocation      = []

for row in X_test:
    tids.append(          int(row[0]))
    tairline.append(          row[1])
    tusername.append(         row[2])
    tretweet_count.append(int(row[3]))
    ttext.append(             row[4])
    ttimestamp.append(        row[5])
    tlocation.append(         row[6])

if file_name_test == 'airline-test-features.csv':
    if not len(tlocation) == 3000:
        print("something went wrong: number of instances in data:",len(tlocation),"does not match original number of instances:", len_X_test)


#balance data
pos = (y_train == 'positive')  
neg = (y_train == 'negative')
neu = (y_train == 'neutral')
print(numpy.sum(pos))
print(numpy.sum(neg))
print(numpy.sum(neu))


        
        
        
#tokenizing text with scikit-learn, see tutorial at: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect        = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(text)
X_test_counts  = count_vect.transform(ttext)

#get the Term Frequencies and perform a downscaling algorithm

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf  = tfidf_transformer.transform(X_test_counts)

#overrepresentation of negatives
#how to balance?

idnr = list(range(0, X_train_tfidf.shape[0]))
idnr = numpy.asarray(idnr)
posnr = idnr[pos]
negnr = idnr[neg]
neunr = idnr[neu]

print(posnr.shape[0],negnr.shape[0],neunr.shape[0])

#get from each vector 1000 random nrs
from numpy.random import choice
rposnr = numpy.random.choice(posnr, size = posnr.shape[0], replace = False)
rnegnr = numpy.random.choice(negnr, size = negnr.shape[0], replace = False)
rneunr = numpy.random.choice(neunr, size = neunr.shape[0], replace = False)

#concatenate nrs
randoms = numpy.concatenate((rposnr,rnegnr,rneunr), axis = 0)


#get the random data
X_train_tfidf = X_train_tfidf[randoms,:]
y_train = y_train[randoms,]
       



#let's fit a naive baysian model that provides a nice baseline for this kind of analysis
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha = 0.9)


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train_tfidf, y_train, cv=10)
print("Accuracy M_NB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


##SGD CLASSIFIER WITH SVM
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE

svmclass = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
scores = cross_val_score(svmclass, X_train_tfidf, y_train, cv=10)
print("Accuracy SGD_SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#using sdg classifer
print(X_train_tfidf.shape)

best_SGD_model = SGDClassifier(penalty="none", class_weight = "balanced", loss="log", random_state=666, )
best_SGD_model.fit(X_train_tfidf,y_train)
rfe = RFE(best_SGD_model, 100)
y_pred = best_SGD_model.predict(X_test_tfidf)

scores = cross_val_score(best_SGD_model, X_train_tfidf, y_train, cv=10)

print("Accuracy best SGD: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


