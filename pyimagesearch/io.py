from keras.utils import np_utils
import numpy as np 
import h5py
import os

class HDF5DatasetWriter:
	def __init__(self, dims, outputPath, dataKey="images",max_label_length=1,
		bufSize=1000):
		if os.path.exists(outputPath):
			raise ValueError("The supplied 'outputPath' already "
				"exists and cannot be overwritten. Manually delete"
				"the file before continuing.", outputPath)
		self.max_label_length = max_label_length

		self.db = h5py.File(outputPath, "w")
		self.data = self.db.create_dataset(dataKey, dims,
			dtype="float")
		self.labels = self.db.create_dataset("labels", (dims[0], self.max_label_length),
			dtype="int")	
		self.bufSize = bufSize
		self.buffer = {"data": [], "labels": []}
		self.idx = 0
	def add(self, rows, labels):
		self.buffer["data"].extend(rows)
		self.buffer["labels"].extend(labels)

		if len(self.buffer["data"]) >= self.bufSize:
			self.flush()

	def flush(self):
		i = self.idx + len(self.buffer["data"])
		self.data[self.idx:i] = self.buffer["data"]
		self.labels[self.idx:i] = self.buffer["labels"]
		self.idx = i
		self.buffer = {"data": [], "labels": []}

	def storeClassLabels(self, classLabels):
		dt = h5py.special_dtype(vlen=str)
		labelSet = self.db.create_dataset("label_names",
			(len(classLabels),), dtype=dt)
		labelSet[:] = classLabels
	def close(self):
		if len(self.buffer["data"]) > 0:
			self.flush()

		self.db.close()
class HDF5DatasetGenerator:
	def __init__(self, dbPath, batchSize, preprocessors=None,
		aug=None, binarize=True, classes=2, max_label_length=1):
		

		self.batchSize = batchSize
		self.preprocessors = preprocessors
		self.aug = aug
		self.binarize = binarize
		self.classes = classes 
		self.max_label_length = max_label_length
		self.db = h5py.File(dbPath)
		self.numImages = self.db["labels"].shape[0]
	def generator(self, passes=np.inf):
		epochs = 0
		while epochs<passes:
			for i in np.arange(0, self.numImages, self.batchSize):
				images = self.db["images"][i:i+self.batchSize]
				labels = self.db["labels"][i:i+self.batchSize]
				

				if self.binarize:
					labels = np_utils.to_categorical(labels,
						self.classes)
				if self.preprocessors is not None:
					procImages = []

					for image in images:
						for p in self.preprocessors:
							image = p.preprocess(image)

						procImages.append(image)

					images = np.array(procImages)

				if self.aug is not None:
					(images, labels) = next(self.aug.flow(images,
						labels, batch_size=self.batchSize))

				input_length = np.ones((self.batchSize, 1)) * 78 #(80 timesteps-2) 
				label_length = np.zeros((self.batchSize, 1))

				for i in range(self.batchSize):
					label_length[i] = self.max_label_length

				inputs = {
				'input': images,
				'label': labels,
				'input_length': input_length,
				'label_length': label_length
				}
				outputs = {'ctc': np.zeros([self.batchSize])} 

				yield(inputs, outputs)

			epochs += 1
	def close(self):
		self.db.close()

