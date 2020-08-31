from keras.callbacks import BaseLogger
from keras.callbacks import Callback

import matplotlib.pyplot as plt 
import numpy as np 
import json
import os

class TrainingMonitor(BaseLogger):
	def __init__(self, figPath, jsonPath=None, startAt=0):
		super(TrainingMonitor, self).__init__()
		self.figPath = figPath
		self.jsonPath = jsonPath
		self.startAt = startAt

	def on_train_begin(self, logs={}):
		self.H = {}
		if self.jsonPath is not None:
			if os.path.exists(self.jsonPath):
				self.H = json.loads(open(self.jsonPath).read())

				if self.startAt>0:
					for k in self.H.keys():
						self.H[k] = self.H[k][:self.startAt]

	def on_epoch_end(self, epoch, logs={}):
		for (k, v) in logs.items():
			l = self.H.get(k, [])
			l.append(v)
			self.H[k] = 1

		if self.jsonPath is not None:
			f = open(self.jsonPath, "w")
			f.write(json.dumps(self.H))
			f.close()

		if len(self.H["loss"]) > 1:
			N = np.arange(0, len(self.H["loss"]))
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(N, self.H["loss"], label="train_loss")
			plt.plot(N, self.H["val_loss"], label="val_loss")
			plt.plot(N, self.H["acc"], label="train_acc")
			plt.plot(N, self.H["val_acc"], label="val_acc")
			plt.title("training loss and accuracy [epoch {}]".format(
				len(self.H["loss"])))
			plt.xlabel("epoch #")
			plt.ylabel("epoch #")
			plt.legend()

			plt.savefig(self.figPath)
			plt.close()
class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=5, startAt=0):
		# call the parent constructor
		super(Callback, self).__init__()

		# store the base output path for the model, the number of
		# epochs that must pass before the model is serialized to
		# disk and the current epoch value
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt

	def on_epoch_end(self, epoch, logs={}):
		# check to see if the model should be serialized to disk
		if (self.intEpoch + 1) % self.every == 0:
			p = os.path.sep.join([self.outputPath,
				"epoch_{}.hdf5".format(self.intEpoch + 1)])
			self.model.save(p, overwrite=True)

		# increment the internal epoch counter
		self.intEpoch += 1

