import sys,os
import numpy as np
from sklearn import mixture
from sklearn.linear_model import LogisticRegression
from functions.MIL2SIL import MIL2SIL

class BOW(object):
        
    def __init__(self):
        self._logistic = None
        self._gauss_mix_model = None
    
    def fit(self,train_bags,train_labels,**kwargs):
        k = kwargs['k']
        covar_type = kwargs['covar_type']
        n_iter = kwargs['n_iter']
        X, Y = MIL2SIL(train_bags,train_labels)
        self._gauss_mix_model=mixture.GaussianMixture(n_components=k,covariance_type=covar_type, init_params='random')
        self._gauss_mix_model.fit(X)
        out_hist = self._gauss_mix_model.predict_proba(X)
        self._logistic = LogisticRegression()
        self._logistic = self._logistic.fit(out_hist,Y)
        
    def predict(self,test_bags):
        n = len(test_bags)
        bags_out_test=[]
        for i in range (0,n):
            sil_bag, _= MIL2SIL(test_bags[i],[0])
            out_test = self._gauss_mix_model.predict_proba(sil_bag)
            out_test = np.mean(out_test,axis=0)
            bags_out_test.append(out_test.reshape(1,len(out_test)))

        bags_out_test = np.vstack(bags_out_test)
        out_predicted = self._logistic.predict(bags_out_test)
    
        return out_predicted 
        
        
        
        