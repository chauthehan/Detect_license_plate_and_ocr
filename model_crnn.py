from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Reshape, Bidirectional, LSTM, Input, Lambda
from keras import backend as K
from config import config

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]  #chiều bây giờ là 98
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class CRNN:
	def build(width, height, depth, classes, training = 1):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		inputs = Input(name='input',shape=inputShape, dtype='float32')

		model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
		model = ELU()(model)
		model = BatchNormalization(axis=chanDim)(model)
		model = (MaxPooling2D(pool_size=(2 ,2)))(model)

		model = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(model)
		model = ELU()(model)
		model = BatchNormalization(axis=chanDim)(model)
		model = MaxPooling2D(pool_size=(1, 2))(model)		

		model = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(model)
		model = ELU()(model)
		model = BatchNormalization(axis=chanDim)(model)


		model = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(model)
		model = ELU()(model)
		model = BatchNormalization(axis=chanDim)(model)
		model = MaxPooling2D(pool_size=(1, 2))(model)
		
		model = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(model)
		model = ELU()(model)
		model = BatchNormalization(axis=chanDim)(model)

		
		model = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(model)
		model = ELU()(model)
		model = BatchNormalization(axis=chanDim)(model)
		model = MaxPooling2D(pool_size=(1, 2))(model)

		model = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal')(model)
		model = ELU()(model)
		model = BatchNormalization(axis=chanDim)
		
		model = Reshape((64, 1280))(model)
		model = Dense(64, activation='relu', kernel_initializer='he_normal',name='dense1')(model)
		
		model = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'))(model)
		model = BatchNormalization(axis=chanDim)(model)

		model = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'))(model)
		model = BatchNormalization(axis=chanDim)(model)

		model = Dense(216, kernel_initializer='he_normal')(model)
		y_pred = Activation("softmax")(model)

		labels = Input(name='label',shape=[20], dtype='float32') 
		input_length = Input(name='input_length',shape=[1], dtype='int64')     # (None, 1)
		label_length = Input(name='label_length',shape=[1], dtype='int64') 

		loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([
			y_pred, labels, input_length, label_length])


		if training==1:
			return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
		else:
			return Model(inputs=[inputs], outputs=[y_pred])

model = CRNN.build(width=160, height=32, depth=1,classes=215)
model.summary()

