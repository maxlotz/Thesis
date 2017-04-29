#listvar="rgb hsv jet parula"
# listvar="hsv hsvnorm"
# for j in $listvar
# do
# 	for i in {0..10}
# 	do
# 		/home/maxlotz/caffe/build/tools/caffe train --solver /data/caffe_models/prototxts/Washington/${j}/solver_$i.prototxt --weights /data/caffe_models/models/bvlc_alexnet.caffemodel 2>&1 | tee /data/caffe_models/logs/alexnet_Washington_${j}_$i.log
# 		python /home/maxlotz/caffe/tools/extra/parse_log.py /data/caffe_models/logs/alexnet_Washington_${j}_$i.log /data/caffe_models/logs/ --delimiter " "
# 		python /home/maxlotz/Thesis/Code/plot_learning_curve.py /data/caffe_models/logs/alexnet_Washington_${j}_$i.log alexnet_Washington_${j}_$i
# 	done
# done

# /home/maxlotz/caffe/build/tools/caffe train --solver /data/caffe_models/prototxts/Washington/solver_join.prototxt --weights /data/caffe_models/models/joinnet.caffemodel 2>&1 | tee /data/caffe_models/logs/joinnetrgb_1.log
# python /home/maxlotz/caffe/tools/extra/parse_log.py /data/caffe_models/logs/joinnetrgb_1.log /data/caffe_models/logs/ --delimiter " "
# python /home/maxlotz/Thesis/Code/plot_learning_curve.py /data/caffe_models/logs/joinnetrgb_1.log joinnetrgb_1
python /home/maxlotz/Thesis/Code/plot_learning_curve.py /data/caffe_models/logs/joinnetrgb_1.log joinnetrgb_1