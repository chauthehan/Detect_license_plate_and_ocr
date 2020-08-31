from pyimagesearch.io import HDF5DatasetWriter
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

def resize(image):
	width = config.WIDTH #160
	height = config.HEIGHT #32
	k1 = width/height
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
	
	return results

imagePaths1 = list(paths.list_images('out/'))
imagePaths2 = list(paths.list_images('out_name/'))

random.shuffle(imagePaths1)
random.shuffle(imagePaths2)
#value in config is value after rotate, so width is height, height is width

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
count = 0

writer = HDF5DatasetWriter((100000, config.HEIGHT, config.WIDTH), 'hdf5/train.hdf5', max_label_length=config.MAX_LENGTH)

for j, imagePath1 in enumerate(imagePaths1):

	image1 = cv2.imread(imagePath1, cv2.IMREAD_GRAYSCALE)
		#get label file path
	dot = imagePath1.rfind('.')
	label_file = imagePath1[:dot] + '.txt'

	if image1.shape[0] != 100:
		count = count +1

		results = resize(image1)
		with open(label_file, 'r') as f:
			line = f.readline().rstrip('\n')
			#print(line)
						
			#convert label 
			char_ids_padded = encode_utf8_string(
							text=line,
							dic=dic,
							length=config.MAX_LENGTH,
							null_char_id=config.NUM_CLASSES)
			#print(char_ids_padded)
			
		f.close()
		writer.add([results], [char_ids_padded])


for j, imagePath2 in enumerate(imagePaths2):
	
	image2 = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)
	if image2.shape[0] != 100:
		count = count + 1

		results = resize(image2)

		name = os.path.split(imagePath2)[-1]
		if '_' in name:
			line = name.split('_')[0]
		else:
			line = name.split('.')[0]
		#print(imagePath2)
		
		#convert label 
		char_ids_padded = encode_utf8_string(
						text=line,
						dic=dic,
						length=config.MAX_LENGTH,
						null_char_id=config.NUM_CLASSES)
		writer.add([results], [char_ids_padded])


for j, imagePath1 in enumerate(imagePaths1):
	for j, imagePath2 in enumerate(imagePaths2):
		if count == 100000:
			exit()

		image1 = cv2.imread(imagePath1, cv2.IMREAD_GRAYSCALE)
		 
		image2 = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)

		#get label file path
		dot = imagePath1.rfind('.')
		label_file = imagePath1[:dot] + '.txt'

		#get label for the 2-nd image
		name = os.path.split(imagePath2)[-1]
		if '_' in name:
			label_2 = name.split('_')[0]
		else:
			dot = name.rfind('.')
			label_2 = name[:dot]
		
		if image1.shape[0] == 100 and image2.shape[0]==100:
			count = count + 1
			image = np.concatenate((image1, image2), axis=1)
			image[:, 255:305] = cv2.blur(image[:,255:305], (40, 40))

			results = resize(image)
			
			with open(label_file, 'r') as f:
				line = f.readline().rstrip('\n')
				
				line = line + ' ' + label_2					
				#print(imagePath1)
				#print(imagePath2)
				char_ids_padded = encode_utf8_string(
								text=line,
								dic=dic,
								length=config.MAX_LENGTH,
								null_char_id=config.NUM_CLASSES)

			f.close()

			writer.add([results], [char_ids_padded])

writer.close()
