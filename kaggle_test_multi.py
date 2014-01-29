#!/usr/bin/env python
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
from glob import glob
import pickle
import numpy as np
import csv


def process(mdl, ds, batch_size=100):
    # This batch size must be evenly divisible into number of total samples!
    mdl.set_batch_size(batch_size)
    X = mdl.get_input_space().make_batch_theano()
    Y = mdl.fprop(X)
    y = T.argmax(Y, axis=1)
    f = function([X], y)
    yhat = []
    for i in xrange(ds.X.shape[0] / batch_size):
        x_arg = ds.X[i * batch_size:(i + 1) * batch_size, :]
        yhat.append(f(x_arg.astype(X.dtype)))
    return np.array(yhat)

tst = pickle.load(open('saved_tst.pkl', 'rb'))
ds = DenseDesignMatrix(X=tst)
clfs = glob('ensemble_clf/*.pkl')
if (len(clfs) % 2) == 0:
    raise AttributeError('Use an odd number of voters to avoid ties!')
mdls = (serial.load(f) for f in clfs)

fname = 'results.csv'
test_size = ds.X.shape[0]
res = np.zeros((len(clfs), test_size), dtype='float32')
for n,mdl in enumerate(mdls):
    res[n, :] = process(mdl, ds, batch_size=50).ravel()
    print "Processing model ",n
yhat = np.mean(res, axis=0)
print yhat.shape
raise ValueError()
converted_results = [['id', 'label']] + [[n + 1, int(x)]
                                         for n, x in enumerate(yhat.ravel())]
with open(fname, 'w') as f:
    csv_f = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
    csv_f.writerows(converted_results)
