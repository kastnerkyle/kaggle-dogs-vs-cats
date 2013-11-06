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
        self.n_classes = 10
        self.extend_train = extend_train
        self.label_names = ["cat", "dog"]

        self.ntrain = 20000
        self.ntest = 5000
        if datapath is not None:
            if datapath[-1] != '/':
                datapath += '/'

        print datapath
        if s == 'train':
            print "Loading training set"
            files = glob(datapath + 'train/*.png')
        elif s == 'valid':
            print "Loading validation set"
            files = glob(datapath + 'train/*.png')
            self.extend_train = False
        elif s == 'test':
            files = glob(datapath + 'test1/*.png')
            self.extend_train = False
        else:
            raise ValueError("Only train, test, and valid data is available")
        assert len(files) > 0, "Unable to read files! Ensure correct datapath."

        # Sort the files so they match the labels
        files = sorted(files,
                       key=lambda x: int(x.split("/")[-1].split(".")[-2]))
        #y = np.array([1. if 'dog' in x.split("/")[-1] else 0. for x in files])
        y = np.array([5. if 'dog' in x.split("/")[-1] else 3. for x in files])

        "Total number of files:", len(files)
        X = np.array([np.array(mpimg.imread(files[0])) for f in files])
        X *= 255.0
        X = X.swapaxes(0, 3)

        if self.one_hot:
            hot = np.zeros((y.shape[0], self.n_classes), dtype='float32')
            for i in xrange(y.shape[0]):
                #print "Filename", files[i], "assigning to", int(y[i])
                hot[i, int(y[i])] = 1.
            y = hot

        if self.extend_train:
            from pylearn2.datasets import cifar10
            ds = cifar10.CIFAR10('train',
                                 toronto_prepro=False,
                                 one_hot=True,
                                 axes=('c', 0, 1, 'b'))
            print "Adding CIFAR10", X.shape
            q = ds.get_topological_view()
            #ci = ds.label_names.index('cat')
            #di = ds.label_names.index('dog')
            #d_indices = np.where(np.argmax(ds.y, axis=1) == di)
            #c_indices = np.where(np.argmax(ds.y, axis=1) == ci)
            #hot = np.zeros((ds.y.shape[0], 2))
            #for i in xrange(hot.shape[0]):
            #    if i in list(d_indices[0]):
            #hot[i, 1] = 1.
            #    elif i in list(c_indices[0]):
            #        hot[i, 0] = 1.
            #y = np.vstack((y, hot))
            X = np.concatenate((X, q), axis=3)
            y = np.vstack((y, ds.y))
            self.ntrain = X.shape[0] - self.ntest
            #dog_cifar_images = q[:,:,:, d_indices].squeeze()
            #cat_cifar_images = q[:,:,:, c_indices].squeeze()
            #X = np.concatenate((X, dog_cifar_images), axis=3)
            #X = np.concatenate((X, cat_cifar_images), axis=3)
            #dog_ext = dog_cifar_images.shape[3]
            #cat_ext = cat_cifar_images.shape[3]
            #hot = np.zeros((dog_ext + cat_ext, 2))
            #for i in xrange(dog_ext):
            #    hot[i, 1] = 1.
            #for i in xrange(cat_ext):
            #    hot[i, 0] = 1.
            #y = np.vstack((y, hot))
            #self.ntrain = self.ntrain + dog_ext + cat_ext
            print "Extended to", X.shape

        if s == 'train' or s == 'valid':
            x_state = np.random.RandomState(42)
            X = X.swapaxes(3, 0)
            x_state.shuffle(X)
            X = X.swapaxes(0, 3)
            y_state = np.random.RandomState(42)
            y_state.shuffle(y)
            if s == 'train':
                X = X[:,:,:, :self.ntrain]
                y = y[:self.ntrain,:]
            else:
                X = X[:,:,:, -self.ntest:]
                y = y[-self.ntest:,:]


        super(kaggle_dogsvscats, self).__init__(y=y,
                                                topo_view=X,
                                                axes=axes)
