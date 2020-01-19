# This is where I'm keeping the Restnet Algorithm
import numpy as np
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Conv2D, Dropout, GlobalAveragePooling2D
from keras.layers import Dense, Sequential

def load_model(algo_type="Resnet50"):
	if algo_type == 'Resnet50':
		bottleneck_features = np.load('../data/bottleneck_features/DogResnet50Data.npz')
		train = bottleneck_features['train']
		valid = bottleneck_features['valid']
		test = bottleneck_features['test']
		data_shape = train.shape[1:]
		return train, valid, test, data_shape
	elif algo_type == 'VGG19':
		bottleneck_features = np.load('/data/bottleneck_features/DogVGG19Data.npz')
		train = bottleneck_features['train']
		valid = bottleneck_features['valid']
		test = bottleneck_features['test']
		data_shape = train.shape[1:]
		return train, valid, test, data_shape
	else:
		print('Algorithm type not found')

def build_model(data_shape):
	model = Sequential()
	model.add(GlobalAveragePooling2D(input_shape=input_shape))
	model.add(Dropout(0.2))
	model.add(Dense(133, activation='softmax'))
	return model

def compile_model(model):
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model

def train_model(model, algo_type):
	checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.{}.hdf5'.format(algo_type, 
		verbose=1, save_best_only=True)

	Resnet50_model.fit(train_Resnet50, train_targets, 
		validation_data=(valid_Resnet50, valid_targets),
		epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

if __name__ == '__main__':
	# Add in logic here
	train, valid, test = load_model()



