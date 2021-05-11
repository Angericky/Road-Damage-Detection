import os 
from natsort import natsorted 
import numpy as np

folders = ["/dev/disk2/zjj/data/abnormal_detection/"]
#path = "Augmented_dataset/"
#path2 = "images/"
path = "Part2/"
path2 = "Part3/"
Names = list()
Names2 = list()

for fold in folders:
	for fo in natsorted(os.listdir(fold+path)):
		if '.' not in fo and 'bak' not in fo:
			print('fo: ', fo)
			for img in natsorted(os.listdir(fold+path+fo)):
				if img.endswith(".xml"):
					Names.append(fold + path + fo + "/" + img)
				if img.endswith(".jpg"):
					Names2.append(fold + path + fo + "/" + img)

for fold in folders:
	# for fo in natsorted(os.listdir(fold+path)):
	for img in natsorted(os.listdir(fold+path2)):
		if img.endswith(".xml"):
			Names.append("../train/" + fold + path2+ img)
		if img.endswith(".jpg"):
			Names2.append("../train/" + fold + path2 + img)

np.savetxt("data/train_aug_annotations.txt",Names,fmt = '%s')
np.savetxt("data/train_aug.txt",Names2,fmt = '%s')

folders = ["/dev/disk2/zjj/data/abnormal_detection/test/"]
#path = "Augmented_dataset/"
#path2 = "images/"
path = "IMG/"
path2 = "XML/"

Names = list()
Names2 = list()

for fold in folders:
	for img in natsorted(os.listdir(fold+path)):
		if img.endswith(".jpg"):
			Names2.append(fold + path + img)

	for img in natsorted(os.listdir(fold+path2)):
		if img.endswith(".xml"):
			Names.append(fold + path2 + img)

np.savetxt("data/test_annotations.txt",Names,fmt = '%s')
np.savetxt("data/test.txt",Names2,fmt = '%s')

