import argparse
from keras.models import load_model
from create_model import CRNN
from config import config
from pyimagesearch.io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
import numpy as np
import cv2
import itertools
import os
import imutils
from imutils import paths 


def fastdecode(y_pred, chars):
    results_str = ""
    confidence = 0.0

    for i,one in enumerate(y_pred[0]):

        if one<config.NUM_CLASSES and (i==0 or (one!=y_pred[0][i-1])):
            results_str+= chars[one]

    return results_str
def decode_label(label, chars):
	results_str = ""
	for i, num in enumerate(label[0]):
		if num != config.NUM_CLASSES:
			results_str += chars[num]
	return results_str 

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
	help = "path to pre-trained model")
ap.add_argument("-i", "--images", 
	help = "path to image folder ")
args = vars(ap.parse_args())

print('loading model...')
model = CRNN.build(width=config.WIDTH, height=config.HEIGHT, depth=1,
		classes=config.NUM_CLASSES, training=0)
model.load_weights(args["model"])
iap = ImageToArrayPreprocessor()
dic = {}
dic[0] = ' '
with open('dic.txt', encoding="utf-8") as dict_file:
	for i, line in enumerate(dict_file):
		if i == 0:
			continue
		(key, value) = line.strip().split('\t')
		dic[int(key)] = value
dict_file.close()

acc = 0
total = 0
list_images = paths.list_images(args["images"])


width = config.WIDTH
height = config.HEIGHT
k1 = width/height

print('predicting...')

for path in list_images:

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    k2 = image.shape[1]/image.shape[0]
    if k2 < k1:		
        resized = imutils.resize(image, height = height)
        zeros = np.zeros((height, width - resized.shape[1]))
        #zeros = zeros + 255
        image = np.concatenate((resized, zeros), axis=1)

    else:
        resized = imutils.resize(image, width = width)
        zeros = np.zeros((height - resized.shape[0], width))
        #zeros = zeros + 255
        image = np.concatenate((resized, zeros), axis=0)

    #image = imutils.rotate_bound(results, 90)
    image = image/255.0
    image = [image]
    image = iap.preprocess(image)  


    predict = model.predict([[image]])	
    predict = np.argmax(predict, axis=2)
    #print(len(predict))		

    res = fastdecode(predict, dic)
    print(res)

