#!/usr/bin/env python
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

clf = pickle.load(open('saved_clf.pkl', 'rb'))
tst = pickle.load(open('saved_tst.pkl', 'rb'))
res = clf.predict(tst)
out = np.vstack((np.array(range(1, res.shape[0] + 1)), res)).astype('int').T
np.savetxt('results.csv', out, header='id,label', fmt='%i,%i', delimiter=',')
print 'Results saved'
