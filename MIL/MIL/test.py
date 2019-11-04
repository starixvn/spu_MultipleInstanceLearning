
#Import Libraries
import sys,os
import numpy as np
from sklearn import metrics
from data import load_data
from sklearn.utils import shuffle
import random as rand
from data import load_data
from sklearn.model_selection import train_test_split
from Algorithms.MIL import MIL
from Algorithms.BOW import BOW
import pickle

#Load Data 
#bags,labels = load_data.load_data('robot')  
pickle_in = open("D:/csv/data.dat","rb")
bags = pickle.load(pickle_in)

pickle_in = open("D:/csv/data_label.txt","rb")
labels = pickle.load(pickle_in)
labels = np.delete(labels, 1)
seed = 60

#Shuffle Data
bags,labels = shuffle(bags, labels, random_state=rand.randint(0, 100))

train_bags, test_bags, train_labels, test_labels = train_test_split(bags, labels, test_size=0.4, random_state=seed)

                            
#Bags Of Words #
                           
bow_classifier = BOW() 
bow_classifier.fit(train_bags, train_labels,k=10,covar_type = 'diag',n_iter = 20)
predictions = bow_classifier.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print('\n BOW accuracy: %.2f%%' % (100 * accuracie))
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)

filename = 'D:/Model/model_BOW.sav'
pickle.dump(bow_classifier, open(filename, 'wb'))
                            
SMILa = MIL() 
SMILa.fit(train_bags, train_labels, type='average')
predictions = SMILa.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print ('\n Accuracy: %.2f%%' % (100 * accuracie))
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)
filename = 'D:/Model/model_MIL_average.sav'
pickle.dump(bow_classifier, open(filename, 'wb'))
                            
SMILe = MIL() 
SMILe.fit(train_bags, train_labels, type='extreme')
predictions = SMILe .predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print ('\n Accuracy: %.2f%%' % (100 * accuracie))
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)
filename = 'D:/Model/model_MIL_extreme.sav'
pickle.dump(bow_classifier, open(filename, 'wb'))

SMILmx = MIL() 
SMILmx.fit(train_bags, train_labels, type='max')
predictions = SMILmx.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print ('\n Accuracy: %.2f%%' % (100 * accuracie))
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)
filename = 'D:/Model/model_MIL_max.sav'
pickle.dump(bow_classifier, open(filename, 'wb'))
                           
SMILmn = MIL() 
SMILmn.fit(train_bags, train_labels, type='min')
predictions = SMILmn.predict(test_bags)
accuracie = np.average(test_labels.T == np.sign(predictions))
print ('\n SimpleMIL accuracy: %.2f%%' % (100 * accuracie))
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1.)
metrics.auc(fpr, tpr)

filename = 'D:/Model/model_MIL_min.sav'
pickle.dump(SMILmn, open(filename, 'wb'))


    

      