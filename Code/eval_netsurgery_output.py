import cv2
import caffe
import numpy as np
import lmdb
from sklearn.utils import shuffle
from caffe.proto import caffe_pb2

model_path = '/data/caffe_models/models/joinnet.caffemodel'
prototxt_path = '/data/caffe_models/prototxts/train_val_join.prototxt'

image_path = '/data/datasets/WashingtonRGBD/rgb/Washington_raw_rgb_stretch_10000.png'
depth_path = '/data/datasets/WashingtonRGBD/depth/Washington_raw_hsv_stretch_10000.png'

net = caffe.Net(prototxt_path, model_path, caffe.TEST)

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    return cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

# adds mean and reshapes data into W H C caffe format
transformer = caffe.io.Transformer({'rgbdata': net.blobs['rgbdata'].data.shape})
transformer.set_transpose('rgbdata', (2,0,1))
transformer2 = caffe.io.Transformer({'ddata': net.blobs['ddata'].data.shape})
transformer2.set_transpose('ddata', (2,0,1))

img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
depth = cv2.imread(depth_path, cv2.IMREAD_COLOR)
depth = transform_img(depth, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

net.blobs['rgbdata'].data[...] = transformer.preprocess('rgbdata', img)
net.blobs['ddata'].data[...] = transformer2.preprocess('ddata', depth)

out = net.forward()
print out