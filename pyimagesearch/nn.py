from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K 
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input
#same: keep the same shape of the input after conv


class ShallowNet:
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		model.add(Conv2D(32, (3,3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model 
class MiniVGGNet:
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1 

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		model.add(Conv2D(32, (3, 3), padding="same",
		input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model
		
class LeNet:
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		#first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5,5), padding="same", 
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		#second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5,5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		#first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		#softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model
class FCHeadNet:
	def build(baseModel, classes, D):
		headModel = baseModel.output
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(D, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)

		headModel = Dense(classes, activation="softmax")(headModel)
		return headModel
class AlexNet:
	def build(width, height, depth, classes, reg=0.0002):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
		model.add(Conv2D(96, (11, 11), strides=(4,4),
			input_shape=inputShape, padding="same",
			kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(256, (5, 5), padding="same",
			kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(384, (3,3), padding="same",
			kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(384, (3,3), padding="same",
			kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		
		model.add(Conv2D(256, (3,3), padding="same",
			kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(4096, kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(4096, kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		
		model.add(Dense(classes, kernel_regularizer=l2(reg)))
		model.add(Activation("softmax"))

		return model 
class MiniGoogleNet:
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		return x
	def inception_module(x, numK1x1, numK3x3, chanDim):
		conv_1x1 = MiniGoogleNet.conv_module(x, numK1x1, 1, 1,
			(1, 1), chanDim)
		conv_3x3 = MiniGoogleNet.conv_module(x, numK3x3, 3, 3,
			(1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
		return x
	def downsample_module(x, K, chanDim):
		conv_3x3 = MiniGoogleNet.conv_module(x, K, 3, 3, (2, 2),
			chanDim, padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2,2))(x)
		x = concatenate([conv_3x3, pool], axis=chanDim)
		return x
	def build(width, height, depth, classes):
		inputShape = (height, width, depth)
		chanDim = -1
		
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
		inputs = Input(shape=inputShape)
		x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1),
			chanDim)
		x = MiniGoogleNet.inception_module(x, 32, 32, chanDim)
		x = MiniGoogleNet.inception_module(x, 32, 48, chanDim)
		x = MiniGoogleNet.downsample_module(x, 80, chanDim)
		x = MiniGoogleNet.inception_module(x, 112, 48, chanDim)
		x = MiniGoogleNet.inception_module(x, 96, 64, chanDim)
		x = MiniGoogleNet.inception_module(x, 80, 80, chanDim)
		x = MiniGoogleNet.inception_module(x, 48, 96, chanDim)
		x = MiniGoogleNet.downsample_module(x, 96, chanDim)

		x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
		x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
		x = AveragePooling2D((7, 7))(x)
		x = Dropout(0.5)(x)
		
		x = Flatten()(x)
		x = Dense(classes)(x)
		x = Activation("softmax")(x)
		
		model = Model(inputs, x, name="googlenet")
		return model

# class DeeperGoogleNet:
# 	def conv_module(x, K, kx, ky, stride, padding="same", chanDim, reg=0.0005,
# 		name=None):
# 		(convName, bnName, actName) = (None, None, None)

# 		if name is not None:
# 			convName = name + '_conv'
# 			bnName = name + '_bn'
# 			actName = name + 'act'

# 		x = Conv2D(K, (kx, ky), stride, kernel_regularizer=l2(reg),
# 		 padding=padding, name = convName)(x)
# 		x = BatchNormalization(axis=chanDim, name=bnName)(x)
# 		x = Activation("relu", name=actName)(x)
# 		return x 
# 	def inception_module(x, num1x1, num3x3Reduce, num3x3,
# 			 num5x5Reduce, num5x5, num1x1Proj, chanDim, stage, reg=0.0005):
# 		first = DeeperGoogleNet.conv_module(x, num1x1, 1, 1, (1, 1),
# 		chanDim, reg=reg, name=stage+'_first')
# 		second = DeeperGoogleNet.conv_module(x, num3x3Reduce, 1, 1, (1, 1),
# 			chanDim, name=stage+"_second1")
# 		second = DeeperGoogleNet.conv_module(second, num3x3, 3, 3, (1, 1),
# 			chanDim, name=stage+"_second2")
# 		third = DeeperGoogleNet.conv_module(x, num5x5Reduce, 1, 1, (1, 1),
# 			chanDim, name=stage+"_third1")
# 		third = DeeperGoogleNet.conv_module(third, num5x5, 5, 5, (1, 1),
# 			chanDim, name=stage+"_third2")
# 		forth = MaxPooling2D(pool_size=(3, 3), strides=(1,1), 
# 			padding="same", name=stage+"_pool")(x)
# 		forth = DeeperGoogleNet.conv_module(forth, num1x1Proj, 1, 1, (1, 1),
# 			chanDim, name=stage+"_forth")
# 		x = concatenate([first, second, third, forth], axis=chanDim,
# 			name=stage+ "_mixed")
# 		return x
# 	def build(width, height, depth, classes, reg=0.0005):
# 		inputShape = (width, height, depth)
# 		chanDim = -1

# 		if K.image_data_format = "channels_first":
# 			inputShape = (depth, width, height)
# 			chanDim = 1

# 		inputs = Input(shape=inputShape)
# 		x = DeeperGoogleNet.conv_module(inputs, 64, 5, 5, (1,1), chanDim,
# 			name="block1")
# 		x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool1",
# 			padding="same")(x)
# 		x = DeeperGoogleNet.conv_module(x, 64, 1, 1, (1, 1), chanDim,
# 			name="block2")
# 		x = DeeperGoogleNet.conv_module(x, 192, 3, 3, (1, 1), chanDim,
# 			name="block3")
# 		x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool2",
# 			padding="same")(x)
# 		x = DeeperGoogleNet.inception_module(x, 64, 96, 128, 16, 32, 32, chanDim,
# 			stage="3a")
# 		x = DeeperGoogleNet.inception_module(x, 128, 128, 192, 32, 96, 64, chanDim,
# 			stage="3b")
# 		x = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name="pool3",
# 			padding="same")(x)
# 		x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16,
# 			48, 64, chanDim, "4a", reg=reg)
# 		x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24,
# 			64, 64, chanDim, "4b", reg=reg)
# 		x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24,
# 			64, 64, chanDim, "4c", reg=reg)
# 		x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32,
# 			64, 64, chanDim, "4d", reg=reg)
# 		x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32,
# 			128, 128, chanDim, "4e", reg=reg)
# 		x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
# 			name="pool4")(x)
# 		x = AveragePooling2D((4,4), name="pool5")(x)
# 		x = Dropout(0.4, name="do")(x)
# 		x = Flatten(name="flatten")(x)
# 		x = Dense(classes, kernel_regularizer=l2(reg),
# 			name="labels")(x)
# 		x = Activation("softmax", name="softmax")(x)
# 		model = Model(inputs, x, name="googlenet")

# 		return model
class ResNet:
	def residual_module(data, K, stride, chanDim, red=False,
	                  reg=0.0001, bnEps=2e-5, bnMom=0.9):
		shortcut = data
		bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
		act1 = Activation("relu")(bn1)
		conv1 = Conv2D(int(K*0.25), (1, 1), use_bias=False,
		               kernel_regularizer=l2(reg))(act1)

		bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
		                         momentum=bnMom)(conv1)
		act2 = Conv2D(int(K*0.25), (3,3), strides=stride, 
		              padding="same", use_bias=False,
		              kernel_regularizer=l2(reg))(bn2)
		bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
		                          momentum=bnMom)(act2)
		act3 = Activation("relu")(bn3)
		conv3 = Conv2D(K, (1,1), use_bias=False,
		               kernel_regularizer=l2(reg))(act3)
		if red:
		  shortcut = Conv2D(K, (1,1), strides=stride,
		                    use_bias=False, kernel_regularizer=l2(reg))(act1)
		x = add([conv3, shortcut])
		return x         
	def build(width, height, depth, classes, stages, filters,
	        reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
		inputShape = (height, width, depth)
		chanDim = -1
		if K.image_data_format() == "channels_first":
		  inputShape = (depth, height, width)
		  chanDim = 1
		inputs = Input(shape=inputShape)
		x = BatchNormalization(axis=chanDim, epsilon=bnEps,
		                       momentum=bnMom)(inputs)
		if dataset == "cifar":
		  x = Conv2D(filters[0], (3,3), use_bias=False,
		             padding="same", kernel_regularizer=l2(reg))(x)
		for i in range(0, len(stages)): 
		  stride = (1, 1) if i == 0 else (2, 2)
		  x = ResNet.residual_module(x, filters[i + 1], stride,
		      chanDim, red=True, bnEps=bnEps, bnMom=bnMom)   
		  for j in range(0, stages[i] - 1):
		    # apply a ResNet module
		    x = ResNet.residual_module(x, filters[i + 1],
		    (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
		x = BatchNormalization(axis=chanDim, epsilon=bnEps,
		                       momentum=bnMom)(x)
		x = Activation("relu")(x)
		x = AveragePooling2D((8,8))(x)
		x = Flatten()(x)
		x = Dense(classes, kernel_regularizer=l2(reg))(x)
		x = Activation("softmax")(x)

		model = Model(inputs, x, name="resnet")
		return model






