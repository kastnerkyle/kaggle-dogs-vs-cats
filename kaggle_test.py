#!/usr/bin/env python
from kaggle_dataset import kaggle_dogsvscats
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
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
        x_arg = ds.get_topological_view(x_arg)
        yhat.append(f(x_arg.astype(X.dtype)))
    return np.array(yhat)


preprocessor = serial.load('kaggle_dogsvscats_preprocessor.pkl')
mdl = serial.load('kaggle_dogsvscats_maxout_zca.pkl')

fname = 'kaggle_dogsvscats_results.csv'
test_size = 12500
sets = 1
res = np.zeros((sets, test_size), dtype='float32')
for n, i in enumerate([test_size * x for x in range(sets)]):
    ds = kaggle_dogsvscats('test',
                            datapath='/home/kkastner/kaggle_data/kaggle-dogs-vs-cats',
                            one_hot=True,
                            axes=('c', 0, 1, 'b'))
    print ds.X.shape
    print "LOADED"
    ds.apply_preprocessor(preprocessor=preprocessor, can_fit=False)
    yhat = process(mdl, ds)
    res[n, :] = yhat.ravel()

converted_results = [['id', 'label']] + [[n + 1, int(x)]
                                         for n, x in enumerate(res.ravel())]
with open(fname, 'w') as f:
    csv_f = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
    csv_f.writerows(converted_results)
