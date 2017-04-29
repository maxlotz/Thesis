#!/bin/bash

filepath=/data/datasets/WashingtonRGBD/lmdbs/
t=train_
m=mean_
bp=.binaryproto
#listvar="rgb/ hsv/ jet/ parula/"
listvar="4ch_norm/"

for j in $listvar
do
	for i in {0..10}
	do
		echo creating $j$m$i$bp from $j$t$i
		/home/maxlotz/caffe/build/tools/compute_image_mean -backend=lmdb $filepath$j$t$i $filepath$j$m$i$bp
	done
done
