import csv
import numpy as np
import pdb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
from DataProcessing import DataSet



def create_model(predictors, target):

	input_dim = predictors.shape[1]
	output_dim = target.shape[0]

	model = Sequential()

	model.add(Dense(1000, activation='relu', input_shape=(input_dim,)))
	# model.add(Dense(1000, activation='relu'))
	# model.add(Dense(1000, activation='relu'))


	model.add(Dense(output_dim))
	model.compile(optimizer='adam',
	              loss='mse',
	              metrics=['mse'])

	early_stopping_monitor = EarlyStopping(patience=3, monitor='val_mean_squared_error')

	filepath="weights.best.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', save_best_only=True, mode='max')

	training = model.fit(predictors, target, verbose=True, batch_size = 5, epochs = 100, validation_split = 0.2,callbacks=[early_stopping_monitor, checkpoint])

	#print("Loss: %s, MSE: %s" %(training.history['val_loss'], training.history['mean_squared_error'] ) )
	#print (training.history['val_loss'])
	mse = training.history['mean_squared_error'][len(training.history['mean_squared_error'])-1]
	# Create the plot
	# plt.plot(training.history['val_loss'], 'r')
	# plt.xlabel('Epochs')
	# plt.ylabel('Validation score')
	# plt.show()

	return model, mse
		

if __name__=='__main__':

	# data = load_data()

	# predictors = create_predictors_targets(data)

	# target = output_layer(data)

	path = '../sample_data/sample1.csv'

	data = DataSet(path, header=True)

	predictors,target = data.training_set()

	model, mse = create_model(predictors, target)

	pred_data = predictors[60,:]

	pred_data = np.reshape(pred_data, (1,129))

	predictions = model.predict(pred_data)

	actual = predictors[predictors.shape[0]-1]
	actual = np.delete(actual, 0)

	plt.plot(predictions[0], 'r', actual, 'b' )
	plt.show()
