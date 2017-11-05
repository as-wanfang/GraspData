# Input pipeline
# ref: http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/

import numpy as np
import glob
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

def read_data_sets(train_dir, one_hot=True):
    train_images_list = []
    train_labels_list = []
    test_images_list = []
    test_labels_list = []
    i = 0
    for npzfile in glob.glob(train_dir+'/*.npz'):
        images = np.load(npzfile)
        perm = np.arange(1000)
        np.random.shuffle(perm)
        images_shuffled = images['image'][perm]
        train_images_list.append(images_shuffled[:900])
        train_labels_list.append(i*np.ones([900,1]).astype(int))
        test_images_list.append(images_shuffled[900:])
        test_labels_list.append(i*np.ones([100,1]).astype(int))
        i = i + 1

    train_images = np.vstack(train_images_list)
    train_labels = np.vstack(train_labels_list)
    test_images = np.vstack(test_images_list)
    test_labels = np.vstack(test_labels_list)

    perm = np.arange(9000)
    np.random.shuffle(perm)
    train_images = train_images[perm]
    train_labels = train_labels[perm]

    train = DataSet(train_images, train_labels, reshape=False, dtype='uint8')
    test = DataSet(test_images, test_labels, reshape=False, dtype='uint8')

    return base.Datasets(train=train, validation=[], test=test)
