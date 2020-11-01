from sklearn.feature_extraction.image import extract_patches_2d
import cv2
import os
import numpy as np
import imutils
from keras.preprocessing.image import img_to_array
class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.inter = inter
	def preprocess(self, image):
		return cv2.resize(image, (self.width, self.height), interpolation = self.inter)
		
class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		self.dataFormat = dataFormat

	def preprocess(self, image):
		return img_to_array(image, data_format=self.dataFormat)

class AspectAwarePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		(h ,w) = image.shape[:2]
		dW = 0 
		dH = 0

		if w<h:
			image = imutils.resize(image, width=self.width,
				inter=self.inter)
			dH = int((image.shape[0] - self.height)/2.0)

		else:
			image = imutils.resize(image, height=self.height,
				inter=self.inter)
			dW = int((image.shape[1] - self.width)/2.0)

		(h, w) = image.shape[:2]
		image = image[dH:h-dH, dW:w-dW]

		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)
class MeanPreprocessor:
	def __init__(self, rMean, gMean, bMean):
		self.rMean = rMean
		self.gMean = gMean
		self.bMean = bMean
	def preprocess(self, image):
		(B, G, R) = cv2.split(image.astype("float32"))

		R -= self.rMean
		G -= self.gMean
		B -= self.bMean
		return cv2.merge([B, G, R])
class PatchPreprocessor:
	def __init__(self, width, height):
		self.width = width
		self.height = height
	def preprocess(self, image):
		return extract_patches_2d(image, (self.height, self.width),
			max_patches=1)[0]
class CropPreprocessor:
	def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.horiz = horiz
		self.inter = inter
	def preprocess(self, image):
		crops = []
		(h, w) = image.shape[:2]
		coords = [
			[0, 0 , self.width, self.height],
			[w-self.width, 0, w, self.height],
			[w-self.width, h-self.height, w, h],
			[0, h-self.height, self.width, h]]
		dW = int(0.5*(w - self.width))
		dH = int(0.5*(h - self.height))
		coords.append([dW, dH, w-dW, h-dH])
		for (startX, startY, endX, endY) in coords:
			crop = image[startY:endY, startX:endX]
			crop = cv2.resize(crop, (self.width, self.height),
				interpolation=self.inter)
			crops.append(crop)
		if self.horiz:
			mirrors = [cv2.flip(c, 1) for c in crops]
			crops.extend(mirrors)
		return np.array(crops)
