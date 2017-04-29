rgblist, hsvlist, parulalist, jetlist, monolist = [], [], [], [], []
datapath = "/data/datasets/WashingtonRGBD"
textpath = "/home/maxlotz/Thesis/Code"

for i in xrange(41877):
	rgblist.append(datapath + "/rgb/Washington_raw_rgb_stretch_" + str(i+1) + ".png")
	hsvlist.append(datapath + "/depth/Washington_raw_hsv_stretch_" + str(i+1) + ".png")
	parulalist.append(datapath + "/depth/Washington_raw_parula_stretch_" + str(i+1) + ".png")
	jetlist.append(datapath + "/depth/Washington_raw_jet_stretch_" + str(i+1) + ".png")
	monolist.append(datapath + "/depth/Washington_raw_mono_stretch_" + str(i+1) + ".png")

with open(textpath + "/Washingtonrgb.txt", 'w') as file:
	file.writelines("%s\n" % item  for item in rgblist)

with open(textpath + "/Washingtonhsv.txt", 'w') as file:
	file.writelines("%s\n" % item  for item in hsvlist)

with open(textpath + "/Washingtonparula.txt", 'w') as file:
	file.writelines("%s\n" % item  for item in parulalist)

with open(textpath + "/Washingtonjet.txt", 'w') as file:
	file.writelines("%s\n" % item  for item in jetlist)

with open(textpath + "/Washingtonmono.txt", 'w') as file:
	file.writelines("%s\n" % item  for item in monolist)