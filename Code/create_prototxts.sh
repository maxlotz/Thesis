#!/bin/bash

listvar="rgb hsv jet parula 4ch"
ch4var="4ch_norm"
protopath=/data/caffe_models/prototxts/Washington/
origtrainpath=/data/caffe_models/prototxts/train_val.prototxt
snaporigpath=/data/caffe_models/models/alexnet_Washington_rgb_1
snappath=/data/caffe_models/models/alexnet_Washington_
meanpath=/data/datasets/WashingtonRGBD/lmdbs/rgb/mean_1.binaryproto
trainpath=/data/datasets/WashingtonRGBD/lmdbs/rgb/train_1
testpath=/data/datasets/WashingtonRGBD/lmdbs/rgb/test_1
lmdbpath=/data/datasets/WashingtonRGBD/lmdbs/

for j in $listvar
do
	for i in {0..10}
	do
		cp /data/caffe_models/prototxts/solver.prototxt $protopath$j/solver_$i.prototxt
		cp $origtrainpath $protopath$j/train_val_$i.prototxt
		sed -i "s#$origtrainpath#$protopath$j/train_val_$i.prototxt#g" $protopath$j/solver_$i.prototxt
		sed -i "s#$snaporigpath#$snappath${j}_$i#g" $protopath$j/solver_$i.prototxt
		sed -i "s#$meanpath#$lmdbpath$j/mean_$i.binaryproto#g" $protopath$j/train_val_$i.prototxt
		sed -i "s#$trainpath#$lmdbpath$j/train_$i#g" $protopath$j/train_val_$i.prototxt
		sed -i "s#$testpath#$lmdbpath$j/test_$i#g" $protopath$j/train_val_$i.prototxt
		#sed -i "s#conv1#conv1_4ch#g" $protopath$j/train_val_$i.prototxt
	done
done

# for j in $ch4var
# do
# 	for i in {0..10}
# 	do
# 		cp /data/caffe_models/prototxts/solver.prototxt $protopath$j/solver_$i.prototxt
# 		cp $origtrainpath $protopath$j/train_val_$i.prototxt
# 		sed -i "s#$origtrainpath#$protopath$j/train_val_$i.prototxt#g" $protopath$j/solver_$i.prototxt
# 		sed -i "s#$snaporigpath#$snappath${j}_$i#g" $protopath$j/solver_$i.prototxt
# 		sed -i "s#$meanpath#$lmdbpath$j/mean_$i.binaryproto#g" $protopath$j/train_val_$i.prototxt
# 		sed -i "s#$trainpath#$lmdbpath$j/train_$i#g" $protopath$j/train_val_$i.prototxt
# 		sed -i "s#$testpath#$lmdbpath$j/test_$i#g" $protopath$j/train_val_$i.prototxt
# 		sed -i "s#conv1#conv1_4ch#g" $protopath$j/train_val_$i.prototxt
# 	done
# done