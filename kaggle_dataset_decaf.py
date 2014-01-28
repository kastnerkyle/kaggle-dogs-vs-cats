#!/usr/bin/env python
from decaf.util import transform
from decaf.scripts import imagenet
import logging
import numpy as np
from glob import glob
import matplotlib.image as mpimg
from random import shuffle
import pickle


def load_and_preprocess(net, imagepath, center_only=False,
                        scale=True, center_size=256):
    image = mpimg.imread(imagepath)
    # first, extract the center_sizexcenter_size center.
    image = transform.scale_and_extract(transform.as_rgb(image), center_size)
    # convert to [0,255] float32
    if scale:
        image = image.astype(np.float32) * 255.
    # Flip the image
    image = image[::-1, :].copy()
    # subtract the mean
    image -= net._data_mean
    # oversample the images
    images = net.oversample(image, center_only)
    return images


def activate(net, im):
    image = load_and_preprocess(net, im, center_only=True)
    # Need to classify to pull features back
    net.classify_direct(image)
    # Activation of all convolutional layers and first fully connected
    feat = net.feature('fc6_cudanet_out')[0]
    return feat


def png_to_np(basedir, fetch_target=False):
    logging.getLogger().setLevel(logging.INFO)
    data_root = '/home/kkastner/decaf_models/'
    net = imagenet.DecafNet(data_root + 'imagenet.decafnet.epoch90',
                            data_root + 'imagenet.decafnet.meta')
    files = glob(basedir + '*.png')
    if fetch_target:
        shuffle(files)
        # Sort the files so they match the labels
        target = np.array([1. if 'dog' in f.split("/")[-1] else 0.
                           for f in files],
                          dtype='float32')
    else:
        #Must sort the files for the test sort to assure order!
        files = sorted(files,
                       key=lambda x: int(x.split("/")[-1].split(".")[-2]))
    feature_info = activate(net, files[0])
    feature_count = feature_info.shape[0]
    feature_dtype = feature_info.dtype
    data = np.zeros((len(files), feature_count), dtype=feature_dtype)
    for n, im in enumerate(files):
        data[n, :] = activate(net, im)
        if n % 1000 == 0:
            print 'Reading in image', n
    if fetch_target:
        return data, target
    else:
        return data

x, y = png_to_np(
    '/home/kkastner/kaggle_data/kaggle-dogs-vs-cats/train/', fetch_target=True)
tst = png_to_np('/home/kkastner/kaggle_data/kaggle-dogs-vs-cats/test1/')
pickle.dump(x, open('saved_x.pkl', 'wb'))
pickle.dump(y, open('saved_y.pkl', 'wb'))
pickle.dump(tst, open('saved_tst.pkl', 'wb'))
