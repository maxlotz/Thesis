depths_path = "/home/max/Desktop/Thesis/depthfiles.txt"
classes_path = "/home/max/Desktop/Thesis/classes.txt"
labels_path = "/home/max/Desktop/Thesis/labels.txt"

labellist = []

with open(classes_path,'r') as file:
	classes = file.read().splitlines()

with open(depths_path,'r') as file:
	depthlist = file.readlines()
	for line in depthlist:
		splitpath = line.strip().split('/')
		labellist.append(classes.index(splitpath[6]))

with open(labels_path, 'w') as file:
	file.writelines("%s\n" % item  for item in labellist)
