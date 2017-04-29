import os
#suppresses non warning or error output to the terminal, must be done before import caffe
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import numpy as np

rgbprototxt = "/data/caffe_models/prototxts/train_val.prototxt"
rgbmodel = "/data/caffe_models/models/alexnet_Washington_rgb_1.caffemodel"

dprototxt = "/data/caffe_models/prototxts/train_val.prototxt"
dmodel = "/data/caffe_models/models/alexnet_Washington_hsv_1.caffemodel"

joinprototxt = "/data/caffe_models/prototxts/train_val_join.prototxt"
joinmodel = "/data/caffe_models/models/joinnet.caffemodel"

origlayers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
rgblayers = ['rgbconv1', 'rgbconv2', 'rgbconv3', 'rgbconv4', 'rgbconv5', 'rgbfc6', 'rgbfc7']
dlayers = ['dconv1', 'dconv2', 'dconv3', 'dconv4', 'dconv5', 'dfc6', 'dfc7']

caffe.set_mode_gpu()

rgbnet = caffe.Net(rgbprototxt, rgbmodel, caffe.TEST)
dnet = caffe.Net(dprototxt, dmodel, caffe.TEST)
joinnet = caffe.Net(joinprototxt, caffe.TEST)

for orig, rgb, d in zip(origlayers, rgblayers, dlayers):
	joinnet.params[rgb][0] = rgbnet.params[orig][0]
	joinnet.params[rgb][1] = rgbnet.params[orig][1]
	joinnet.params[d][0] = dnet.params[orig][0]
	joinnet.params[d][1] = dnet.params[orig][1]

if os.path.isfile(joinmodel):
	os.remove(joinmodel)

joinnet.save(joinmodel)

for orig, rgb, d in zip(origlayers, rgblayers, dlayers):
	assert(np.array_equal(joinnet.params[rgb][0].data, rgbnet.params[orig][0].data))
	assert(np.array_equal(joinnet.params[rgb][1].data, rgbnet.params[orig][1].data))
	assert(np.array_equal(joinnet.params[d][0].data, dnet.params[orig][0].data))
	assert(np.array_equal(joinnet.params[d][1].data, dnet.params[orig][1].data))

print "all weights transferrred successfully"