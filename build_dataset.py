from pyimagesearch.io import HDF5DatasetWriter
#from pyimagesearch.io import HDF5DatasetWriter
import numpy as np 
import argparse
from imutils import paths
import cv2
import os
import time
import imutils
import random
from tqdm import tqdm
import gc 
from config import config

def encode_utf8_string(text, length, dic, null_char_id):
    char_ids_padded = [null_char_id]*length
    #char_ids_unpadded = [null_char_id]*len(text)
    for i in range(len(text)):
        hash_id = dic[text[i]]
        char_ids_padded[i] = hash_id
        #char_ids_unpadded[i] = hash_id
    return char_ids_padded


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--dataset", help="path to dataset")

args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

random.shuffle(imagePaths)

#value in config is value after rotate, so width is height, height is width
width = config.WIDTH #160
height = config.HEIGHT #32
k1 = width/height

(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])


writer = HDF5DatasetWriter((20000, config.HEIGHT, config.WIDTH), 'hdf5/val.hdf5', max_label_length=config.MAX_LENGTH)
for j, imagePath in tqdm(enumerate(imagePaths)):
	imagePath2 = ''
	for k in imagePath:
		if k != '\\':
			imagePath2 += k
	#print(imagePath2)
	# imagePath = str(imagePath)
	
	image = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)	
	
	k2 = image.shape[1]/image.shape[0]
	if k2 < k1:		
		resized = imutils.resize(image, height = height)
		zeros = np.zeros((height, width - resized.shape[1]))
		#zeros = zeros + 255
		results = np.concatenate((resized, zeros), axis=1)

	else:
		resized = imutils.resize(image, width = width)
		zeros = np.zeros((height - resized.shape[0], width))
		#zeros = zeros + 255
		results = np.concatenate((resized, zeros), axis=0)
	# cv2.imshow('',results)
	# cv2.imwrite('re.jpg', results)
	# cv2.waitKey(0)

	#get the dictionary
	dic = {}
	dic[" "] = 0
	with open('dic.txt', encoding="utf-8") as dict_file:
	    for i, line in enumerate(dict_file):
	        if i == 0:
	            continue

	        (key, value) = line.strip().split('\t')
	        dic[value] = int(key)
	dict_file.close()

    #convert label     
	for l, char in enumerate(imagePath2):
		if char == '.':
			dot = l
	label = imagePath2[:dot] + '.txt'

	with open(label, 'r') as f:
		for line in f:
			print(line)
			char_ids_padded = encode_utf8_string(
                            text=line,
                            dic=dic,
                            length=config.MAX_LENGTH,
                            null_char_id=config.NUM_CLASSES)
	f.close()

	writer.add([results], [char_ids_padded])
	
writer.close()
