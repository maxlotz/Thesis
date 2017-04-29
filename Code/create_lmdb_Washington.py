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

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    return cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tobytes())

#LOCATION YOU WISH TO SAVE LMDBS IN
lmdb_path = '/data/datasets/WashingtonRGBD/lmdbs/'

#Path of my files containing the lists of all files
listspath = '/home/maxlotz/Thesis/file_lists/Washington'
#file_lists = ['rgb', 'hsv', 'parula','jet']
file_lists = ['hsvnorm']
idxpath = '/home/maxlotz/Thesis/trainval_indexes/'

# gets list of labels
with open('/home/maxlotz/Thesis/file_lists/labels.txt', 'r') as f:
    labellist = f.read().splitlines()
labellist = np.array(map(int, labellist))

# gets lists of all file names
allfiles = []
for encoding in file_lists:
    with open(listspath + encoding + '.txt', 'r') as f:
        allfiles.append(np.array(f.read().splitlines()))

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

# gets list of files corresponding to each training and test split
filestrain = []
filestest = []
for idx, files in enumerate(file_lists):
    filestrain.append([])
    filestest.append([])
    for i in xrange(11):
        filestrain[idx].append(allfiles[idx][trainidx[i]])
        filestest[idx].append(allfiles[idx][testidx[i]])

for idx, files in enumerate(file_lists):
    for i in xrange(11):
        # gives directory names
        lmdbtrain = lmdb_path + files + '/train_' + str(i)
        lmdbtest = lmdb_path + files + '/test_' + str(i)
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
            for in_idx, img_path in enumerate(filestrain[idx][i]):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = transform_img(img)
                label = labeltrain[i][in_idx]
                datum = make_datum(img, label)
                in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
                #print '{:0>5d}'.format(in_idx) + ':' + img_path
        in_db.close()

        # creates test lmdbs
        print '\nCreating ' + lmdbtest
        in_db = lmdb.open(lmdbtest, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            for in_idx, img_path in enumerate(filestest[idx][i]):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = transform_img(img)
                label = labeltest[i][in_idx]
                datum = make_datum(img, label)
                in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
                #print '{:0>5d}'.format(in_idx) + ':' + img_path
        in_db.close()

print '\nFinished processing all images'

#TO CREATE MEAN FILE
#/home/maxlotz/caffe/build/tools/compute_image_mean -backend=lmdb /data/datasets/WashingtonRGBD/lmdbs/parula/train_10 /data/datasets/WashingtonRGBD/lmdbs/parula/mean_10.binaryproto
