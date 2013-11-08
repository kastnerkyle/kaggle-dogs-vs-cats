#!/usr/bin/env python
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from glob import glob
import matplotlib.image as mpimg


class kaggle_dogsvscats(DenseDesignMatrix):

    def __init__(self, s, one_hot=False,
                 datapath=None,
                 extend_train=True,
                 axes=('c', 0, 1, 'b')):
        self.img_shape = (3, 32, 32)
        self.img_size = np.prod(self.img_shape)
        self.one_hot = one_hot
        self.n_classes = 2
        self.extend_train = extend_train
        self.label_names = ["cat", "dog"]

        from pylearn2.datasets import cifar10
        ds = cifar10.CIFAR10('train',
                             toronto_prepro=False,
                             one_hot=True,
                             axes=('c', 0, 1, 'b'))
        q = ds.get_topological_view()
        ci = ds.label_names.index('cat')
        di = ds.label_names.index('dog')
        d_indices = np.where(np.argmax(ds.y, axis=1) == di)
        c_indices = np.where(np.argmax(ds.y, axis=1) == ci)
        dog_cifar_images = q[:,:,:, d_indices].squeeze()
        cat_cifar_images = q[:,:,:, c_indices].squeeze()
        X = dog_cifar_images
        X = np.concatenate((X, cat_cifar_images), axis=3)
        dog_ext = dog_cifar_images.shape[3]
        cat_ext = cat_cifar_images.shape[3]
        hot = np.zeros((dog_ext + cat_ext, self.n_classes))
        for i in xrange(dog_ext):
            hot[i, 1] = 1.
        for i in xrange(dog_ext, dog_ext + cat_ext):
            hot[i, 0] = 1.
        y = hot
        self.ntrain = dog_ext + cat_ext

        if s == 'train' or s == 'valid':
            state = np.random.RandomState(42)
            X = X.swapaxes(3, 0)
            state.shuffle(X)
            X = X.swapaxes(0, 3)
            state = np.random.RandomState(42)
            state.shuffle(y)
            if s == 'train':
                X = X[:,:,:, :self.ntrain]
                y = y[:self.ntrain,:]
            else:
                X = X[:,:,:, -self.ntest:]
                y = y[-self.ntest:,:]

        def saveexample(i):
            t = np.argmax(y, axis=1)[i]
            d = {0: 'cat',
                 1: 'dog'}
            mpimg.imsave(d[t]+`i`+'.png',X[:,:,:,i].swapaxes(0,2))

        for i in xrange(20):
            saveexample(i)

        super(kaggle_dogsvscats, self).__init__(y=y,
                                                topo_view=X,
                                                axes=axes)
