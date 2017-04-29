'''
Title           :plot_learning_curve.py
Description     :This script generates learning curves for caffe models
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160619
version         :0.1
usage           :python plot_learning_curve.py model_1_train.log ./caffe_model_1_learning_curve.png
python_version  :2.7.11
'''

# edited to suit my purposes

import os
import sys
import subprocess
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

plt.style.use('ggplot')


caffe_path = '/home/maxlotz/caffe/'
model_log_path = sys.argv[1]
learning_curve_name = sys.argv[2]
learning_curve_path = '/home/maxlotz/Thesis/Figs/' + learning_curve_name + '.png'

#Get directory where the model logs is saved, and move to it
model_log_dir_path = os.path.dirname(model_log_path)
os.chdir(model_log_dir_path)

'''
Generating training and test logs
'''
#Parsing training/validation logs
#command = caffe_path + 'tools/extra/parse_log.sh ' + model_log_path
#process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
#process.wait()
#Read training and test logs
train_log_path = model_log_path + '.train'
test_log_path = model_log_path + '.test'

train_log = pd.read_csv(train_log_path, delim_whitespace=True)
test_log = pd.read_csv(test_log_path, delim_whitespace=True)

# keep model with highest test accuracy and rename it, remove all other models

idx = np.argmax(test_log.values[:,3])
max_accuracy = test_log.values[idx,3]

print test_log
print idx
print max_accuracy
'''
for i in xrange(51):
	modelname = '/data/caffe_models/models/' + learning_curve_name + '_iter_' + str(i*100)
	solver = modelname + '.solverstate'
	caffemodel = modelname + '.caffemodel'
	if i!=idx:
		if os.path.exists(solver):
			os.remove(solver)
		if os.path.exists(caffemodel):
			os.remove(caffemodel)
	else:
		if os.path.exists(solver):
			os.rename(solver, '/data/caffe_models/models/' + learning_curve_name +'.solverstate')
		if os.path.exists(caffemodel):
			os.rename(caffemodel, '/data/caffe_models/models/' + learning_curve_name + '.caffemodel')

with open('/home/maxlotz/Thesis/Accuracies.txt', 'a') as file:
	file.write(learning_curve_name + "\t" + str(max_accuracy) + "\t" + str(idx) + "\n")
'''

fig, ax1 = plt.subplots()

#Plotting training and test losses

train_loss, = ax1.plot(train_log['NumIters'], train_log['loss'], color='red',  alpha=.5)
test_loss, = ax1.plot(test_log['NumIters'], test_log['loss'], linewidth=2, color='green')
ax1.set_ylim(ymin=0, ymax=3)
ax1.set_xlabel('Iterations', fontsize=15)
ax1.set_ylabel('Loss', fontsize=15)
ax1.tick_params(labelsize=15)

#Plotting test accuracy
ax2 = ax1.twinx()
test_accuracy, = ax2.plot(test_log['NumIters'], test_log['accuracy'], linewidth=2, color='blue')
ax2.set_ylim(ymin=0, ymax=1)
ax2.set_ylabel('Accuracy', fontsize=15)
ax2.tick_params(labelsize=15)
#Adding legend
plt.legend([train_loss, test_loss, test_accuracy], ['Training Loss', 'Test Loss', 'Test Accuracy'],  bbox_to_anchor=(1, 0.8))
plt.title('Training Curve', fontsize=18)
#Saving learning curve
plt.savefig(learning_curve_path)
#plt.show()

'''
Deleting training and test logs
'''
'''
command = 'rm ' + train_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()

command = command = 'rm ' + test_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()
'''