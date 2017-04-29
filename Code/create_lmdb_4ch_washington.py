'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

# I have modified Adil Moujahid's file to suit my purposes

import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
from sklearn.utils import shuffle
import lmdb
import os
import shutil

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, depth, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    imout = np.zeros([img.shape[0], img.shape[1], 4], dtype=np.uint8)
    imout[:,:,:3] = img
    imout[:,:,3] = depth
    imout = cv2.resize(imout, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return imout

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=4,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tobytes())

#LOCATION YOU WISH TO SAVE LMDBS IN
lmdb_path = '/data/datasets/WashingtonRGBD/lmdbs/4ch_norm'
idxpath = '/home/maxlotz/Thesis/trainval_indexes/'
listspath = '/home/maxlotz/Thesis/file_lists'

with open(listspath + '/Washingtonrgb.txt', 'r') as file:
    rgblist = file.read().splitlines()
    rgblist = np.array(rgblist)

with open(listspath + '/Washingtonmononorm.txt', 'r') as file:
    monolist = file.read().splitlines()
    monolist = np.array(monolist)

# gets list of labels
with open('/home/maxlotz/Thesis/file_lists/labels.txt', 'r') as f:
    labellist = f.read().splitlines()
    labellist = np.array(map(int, labellist))

# gets list of indexes for all test and training set splits
testidx = []
trainidx = []
for i in xrange(11):
    with open(idxpath + 'trainidx_' + str(i) + '.txt', 'r') as f:
        train_ = map(int,f.read().splitlines())
        # indexes created in matlab (uses indexing starting at 1)
        train_ = np.array(train_) -1
        trainidx.append(train_)
    with open(idxpath + 'testidx_' + str(i) + '.txt', 'r') as f:
        test_ = map(int,f.read().splitlines())
        test_ = np.array(test_) -1
        testidx.append(test_)

# gets list of labels corresponding to each training and test split
labeltest = []
labeltrain = []
for i in xrange(11):
    labeltrain.append(labellist[trainidx[i]])
    labeltest.append(labellist[testidx[i]])

rgbtrain = []
rgbtest = []
monotrain = []
monotest = []
for i in xrange(11):
    rgbtrain.append(rgblist[trainidx[i]])
    rgbtest.append(rgblist[testidx[i]])
    monotrain.append(monolist[trainidx[i]])
    monotest.append(monolist[testidx[i]])

sz = len(rgblist)
for i in xrange(11):
    # gives directory names
    lmdbtrain = lmdb_path + '/train_' + str(i)
    lmdbtest = lmdb_path + '/test_' + str(i)
    # deletes directories if existent, then creates these directories
    if os.path.exists(lmdbtrain):
        shutil.rmtree(lmdbtrain)
    if os.path.exists(lmdbtest):
        shutil.rmtree(lmdbtest)
    os.makedirs(lmdbtrain)
    os.makedirs(lmdbtest)

    # creates training lmdbs
    print '\nCreating ' + lmdbtrain
    in_db = lmdb.open(lmdbtrain, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, (img_path, depth_path) in enumerate(zip(rgbtrain[i],monotrain[i])):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            img = transform_img(img, depth)
            label = labeltrain[i][in_idx]
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
            #print '{:0>5d}'.format(in_idx) + ':' + img_path
    in_db.close()

    # creates test lmdbs
    print '\nCreating ' + lmdbtest
    in_db = lmdb.open(lmdbtest, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, (img_path, depth_path) in enumerate(zip(rgbtest[i],monotest[i])):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            img = transform_img(img, depth)
            label = labeltest[i][in_idx]
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
            #print '{:0>5d}'.format(in_idx) + ':' + img_path
    in_db.close()

print '\nFinished processing all images'

#TO CREATE MEAN FILE
#/home/maxlotz/caffe/build/tools/compute_image_mean -backend=lmdb /data/datasets/WashingtonRGBD/lmdbs/4ch/train /data/datasets/WashingtonRGBD/lmdbs/4ch/mean.binaryproto
