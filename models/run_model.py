# This is where I'm keeping the Restnet Algorithm
import numpy as np
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Conv2D, Dropout, GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import sys

def load_model(algo_type):
	'''
	Input:
	1. algo_type: 'VGG19' or 'Resnet50' are the only options for now
	Output relevant data imported
	1. train: training data
	'''
	if algo_type == 'Resnet50':
		bottleneck_features = np.load('../data/bottleneck_features/DogResnet50Data.npz')
		train = bottleneck_features['train']
		valid = bottleneck_features['valid']
		test = bottleneck_features['test']
		data_shape = train.shape[1:]
		return train, valid, test, data_shape
	elif algo_type == 'VGG19':
		bottleneck_features = np.load('../data/bottleneck_features/DogVGG19Data.npz')
		train = bottleneck_features['train']
		valid = bottleneck_features['valid']
		test = bottleneck_features['test']
		data_shape = train.shape[1:]
		return train, valid, test, data_shape
	else:
		print('Algorithm type not found')

def build_model(data_shape):
	'''
	Pretty simple model here but it's accurate!
	Input: shape of the last 3 values of the input vector data

	Output: model
	'''
	model = Sequential()
	model.add(GlobalAveragePooling2D(input_shape=data_shape))
	model.add(Dropout(0.2))
	model.add(Dense(133, activation='softmax'))
	return model

def compile_model(model):
	'''
	Input: model
	Output: compiled model
	'''
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model

def train_model(model, algo_type, train, valid):
	'''
	Input:
	1. Model
	2. algo_type: currently 'Resnet50' or 'VGG19'
	3. train data
	4. valid data

	Output: 
	fit model
	'''
	checkpointer = ModelCheckpoint(filepath='../models/saved_models/weights.best.{}.hdf5'.format(algo_type), 
		verbose=1, save_best_only=True)

	model.fit(train, train_targets, 
		validation_data=(valid, valid_targets),
		epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
	return model

def get_model(algo_type="Resnet50", retrain=False):
	'''
	This essentially acts like a main for this python file. it runs through all of the products.
	Notes: I offer the retrain option here. But it's much faster and I default to not having this an option.
	'''
	train, valid, test, data_shape = load_model(algo_type)
	model = build_model(data_shape)
	model = compile_model(model)
	if retrain==True:
		model = train_model(model, algo_type, train, valid)
	saved_model = '../models/saved_models/weights.best.{}.hdf5'.format(algo_type)
	model.load_weights(saved_model)
	return model



