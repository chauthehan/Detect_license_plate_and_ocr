#import matplotlib
#matplotlib.use("Agg")

from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.models import load_model
from create_model import CRNN
from config import config
import argparse
import keras.backend as K
import os


ap = argparse.ArgumentParser()

ap.add_argument("-c", "--checkpoints",
	help="path to output checkpoints")
ap.add_argument("-m", "--model",
	help="path to model")
ap.add_argument("-s", "--start_epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())

trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
	rescale=1/255.0, fill_mode='nearest')
valAug = ImageDataGenerator(rescale=1/255.0)
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
	aug=trainAug, preprocessors=[iap], binarize=False, classes=config.NUM_CLASSES,max_label_length=config.MAX_LENGTH)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
	aug=valAug, preprocessors=[iap], binarize=False, classes=config.NUM_CLASSES,max_label_length=config.MAX_LENGTH)


adam = Adam(lr=0.001)
if args['model'] is None:
	print("[info] compiling model..")
	model = CRNN.build(width=config.WIDTH, height=config.HEIGHT, depth=1,
		classes=config.NUM_CLASSES)
	#print(model.summary())
	model.compile(loss={'ctc': lambda y_true, y_pred:y_pred}, optimizer=adam)
else:
	print("[info] loading {}..".format(args["model"]))
	model = CRNN.build(width=config.WIDTH, height=config.HEIGHT, depth=1,
		classes=config.NUM_CLASSES)
	model.load_weights(args["model"])

	model.compile(loss={'ctc': lambda y_true, y_pred:y_pred}, optimizer=adam)


callbacks = [
	EpochCheckpoint(args["checkpoints"], every=5,
		startAt=args["start_epoch"])]

model.fit(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages//config.BATCH_SIZE,
	epochs=30,
	max_queue_size=config.BATCH_SIZE*2,
	callbacks=callbacks,
	verbose=1)
trainGen.close()
valGen.close()
