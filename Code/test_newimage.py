import cv2
import caffe
import numpy as np
import lmdb
from sklearn.utils import shuffle
from caffe.proto import caffe_pb2

model_path = '/data/caffe_models/models/alexnet_transfer_iter_10000.caffemodel'
prototxt_path = '/data/caffe_models/prototxts/deploy.prototxt'
mean_path = '/data/datasets/WashingtonRGBD/lmdbs/rgb/mean.binaryproto'
image_path = '/home/maxlotz/Thesis/bell_pepper.jpg'
label_path = '/home/maxlotz/Thesis/Code/labels.txt'
class_path = '/home/maxlotz/Thesis/Code/classes.txt'

net = caffe.Net(prototxt_path, model_path, caffe.TEST)

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    return cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

mean_blob = caffe_pb2.BlobProto()
with open(mean_path) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
	(mean_blob.channels, mean_blob.height, mean_blob.width))

# adds mean and reshapes data into W H C caffe format
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

with open(class_path,'r') as file:
	class_list = file.read().splitlines()


img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
net.blobs['data'].data[...] = transformer.preprocess('data', img)
out = net.forward()
out = out['prob']
pred_class =  class_list[np.argmax(out)]
print "predicted:" + "\t" + str(pred_class)
cv2.imshow("image", img);
cv2.waitKey();
count += 1
