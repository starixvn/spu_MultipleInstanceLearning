import pickle
import numpy as np
from sklearn import metrics
from data import load_data
from matplotlib import pyplot
#Load Data 
bags,labels = load_data.load_data('robot_test')  

filename = 'D:/Model/model_BOW.sav'

pickle_in = open(filename,"rb")
model = pickle.load(pickle_in)

y_pre=model.predict(bags)

accurancy = np.average(labels.T == np.sign(y_pre))
print ('\n BOW Accuracy: %.2f%%' % (100 * accurancy))
fpr, tpr, thresholds = metrics.roc_curve(labels, y_pre, pos_label=1.)
metrics.auc(fpr, tpr)


filename = 'D:/Model/model_MIL_average.sav'
pickle_in = open(filename,"rb")
model = pickle.load(pickle_in)

y_pre=model.predict(bags)

accurancy = np.average(labels.T == np.sign(y_pre))
print ('\n model_MIL_average Accuracy: %.2f%%' % (100 * accurancy))

filename = 'D:/Model/model_MIL_extreme.sav'
pickle_in = open(filename,"rb")
model = pickle.load(pickle_in)

y_pre=model.predict(bags)

accurancy = np.average(labels.T == np.sign(y_pre))
print ('\n model_MIL_extreme Accuracy: %.2f%%' % (100 * accurancy))


filename = 'D:/Model/model_MIL_max.sav'
pickle_in = open(filename,"rb")
model = pickle.load(pickle_in)

y_pre=model.predict(bags)

accurancy = np.average(labels.T == np.sign(y_pre))
print ('\n model_MIL_max Accuracy: %.2f%%' % (100 * accurancy))

filename = 'D:/Model/model_MIL_min.sav'

pickle_in = open(filename,"rb")
model = pickle.load(pickle_in)

y_pre=model.predict(bags)

accurancy = np.average(labels.T == np.sign(y_pre))
print ('\n model_MIL_min Accuracy: %.2f%%' % (100 * accurancy))




