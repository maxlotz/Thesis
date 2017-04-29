import numpy as np
import lmdb
import caffe
import cv2

from random import randint

lmdbpath = '/data/datasets/WashingtonRGBD/lmdbs/rgb/test_1'
classes_path = '/home/maxlotz/Thesis/file_lists/classes.txt'

with open(classes_path,'r') as file:
	class_list = file.read().splitlines()

lmdb_env = lmdb.open(lmdbpath,readonly=True)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
	datum.ParseFromString(value)
	flat_x = np.fromstring(datum.data, dtype=np.uint8)
	data = flat_x.reshape(datum.channels, datum.height, datum.width)
	#rolling it back to the H W C format for imshow, caffe uses C H W
	data = np.rollaxis(data,2)
	data = np.rollaxis(data,2)
	label = datum.label
	print class_list[label] + "\t" + str(key)
	cv2.imshow("image", data);
	cv2.waitKey();
	#print str(key) + "\t" + str(data.size)

# /home/maxlotz/caffe/build/tools/caffe train --solver /data/caffe_models/prototxts/solver.prototxt --weights /data/caffe_models/models/bvlc_alexnet.caffemodel 2>&1 | tee /data/caffe_models/logs/alexnet_transfer_rgb_1.log
# python /home/maxlotz/caffe/tools/extra/parse_log.py /data/caffe_models/logs/alexnet.log /data/caffe_models/logs/ --delimiter " "