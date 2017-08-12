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
import time



def create_model(predictors, target):

	input_dim = predictors.shape[1]
	output_dim = target.shape[1]

	model = Sequential()

	model.add(Dense(200, activation='relu', input_shape=(input_dim,)))
	#model.add(Dense(8000, activation='relu'))
	# model.add(Dense(1000, activation='relu'))


	model.add(Dense(output_dim))
	model.compile(optimizer='adam',
	              loss='mse',
	              metrics=['mse'])

	early_stopping_monitor = EarlyStopping(patience=2, monitor='val_loss')

	filepath="weights.best.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', save_best_only=True, mode='max')

	training = model.fit(predictors, target, verbose=True, batch_size = 10, epochs = 10, validation_split = 0.2,callbacks=[early_stopping_monitor, checkpoint])

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


	path = '../sample_data/sample3.csv'

	data = DataSet(path, header=True)

	data.process_time_data(days_back=60, extrapolate_method='connect_points')

	predictors,target = data.training_set(output_type='regular')

	start = time.time()

	actual = predictors[predictors.shape[0]-1]

	model, mse = create_model(predictors, target)

	end = time.time()

	print ("%0.2f" %((end-start)/60))

	while True:

		row = int(input("Enter number between 0 and %s" %predictors.shape[0]))

		pred_data = predictors[row,:]

		pred_data = np.reshape(pred_data, (1,predictors.shape[0]))

		predictions = model.predict(pred_data)

		plt.plot(predictions[0], 'r', actual, 'b' )
		plt.show()

		time.sleep(10)

		plt.close()
