import csv
import numpy as np
from .algebra_functions import *
import pdb
import keras.utils

def map_section(section):

	section_map = {182157:1,
			181979:0,
			181981:2}

	return section_map[section]


class DataSet:

	def __init__(self, path, data_matrix):

		self.time_processed = False

		self.time_training_set = False

		self.processed_matrix = None

		self.training_inputs = None

		self.training_outputs = None

		# with open(path, 'rU') as data_file:

		# 	reader = csv.reader(data_file)

		# 	if header:
		# 		next(reader,None)

		# 	data = np.array([row for row in reader])


		self.raw_data_matrix = data_matrix


	def process_time_data(self, days_back=90, extrapolate_method='connect_points'):

		length = days_back * 6 * 24

		# Create array of times
		time_array = np.arange(length)

		raw_length = int(self.raw_data_matrix.shape[0])

		processed_length = int(time_array.shape[0])

		self.new_unprocessed_matrix = np.zeros([processed_length,2])

		self.new_unprocessed_matrix[:,0] = time_array


		# Transform raw times
		for row in self.raw_data_matrix:

			time = row[0]

			y = float(row[1])

			new_time = int(round(float(time)*24.0*6.0))
			
			index = 0

			# Find and replace in the new matrix
			for row in self.new_unprocessed_matrix:

				if new_time==row[0]:
					
					self.new_unprocessed_matrix[index,1] = y

				index+=1

		if extrapolate_method=='connect_points':

			self.processed_time_series = connect_points(self.new_unprocessed_matrix)
			self.processed = True


		last_point = int(round(days_back * 24 * 6))

		self.processed_time_series = self.processed_time_series[0:last_point, :]



	def full_series_training_set(self, output_type='regular'):

		# If time series not yet processed - then process

		if self.time_processed == False:
			self.process_time_data()


		# Create Input Matrix for Training Set

		n_data_points = self.processed_time_series.shape[0]

		time_array = self.processed_time_series[:,0]
		y_array = self.processed_time_series[:,1]

		self.full_series_training_outputs = np.array([y_array for i in range(len(y_array))])

		input_matrix = np.zeros([n_data_points,n_data_points])

		output_matrix = np.zeros([n_data_points,n_data_points])

		temp_input_array = np.zeros(n_data_points)

		temp_output_array = y_array


		for x in range(n_data_points):

			current_y = y_array[x]

			temp_input_array[x] = y_array[x]

			temp_output_array[x] = 0

			input_matrix[x] = temp_input_array

			output_matrix[x] = temp_output_array


		self.full_series_training_inputs = input_matrix

		if output_type=="zeros":

			self.full_series_training_inputs = output_matrix

		self.time_training_set = True

		return self.full_series_training_inputs,self.full_series_training_outputs 



	def sliding_window_training_set(self, days_back=30, days_ahead=30):

		if not self.time_processed:

			self.process_time_data()


		len_back_sliding_window = days_back*6*24
		len_ahead_sliding_window = days_ahead*6*24

		time_array = self.processed_time_series[:,0]
		y_array = self.processed_time_series[:,1]

		# Create matrix for training set - length of time series, and width of sliding window
		input_matrix = np.zeros([time_array.shape[0],len_back_sliding_window+2])

		# Input first column as current time
		input_matrix[:,0] = np.reshape(time_array, (1,time_array.shape[0]))

		# Input second column current price
		input_matrix[:,1] = np.reshape(y_array, (1,y_array.shape[0]))

		# Create output matrix - sequence for future prices
		output_matrix = np.zeros([time_array.shape[0],len_ahead_sliding_window])

		current_row = 0

		for x in self.processed_time_series:

			current_time = x[0]
			current_y = x[1]

			# Loop through each days back and find difference in y
			for k in range(len_back_sliding_window):

				try:
					y_to_compare = y_array[current_row+k+1]
				except:
					y_to_compare = current_y


				# Find difference and input in matrix (starting in third column)
				input_matrix[current_row][k+2] = y_to_compare - current_y


			# Loop ahead for outputs to find prices in future
			for k in range(len_ahead_sliding_window):

				# Check if we have a time ahead
				if current_row-k-1 < 0:
					
					y_to_compare = current_y
					#pdb.set_trace()
					#print ("row: %s, k: %s, y_to_compare: %s" %(current_row,k, y_to_compare))
				else:
					y_to_compare = y_array[current_row-k-1]
					


				# Find difference and put into matrix (starting in third column)
				output_matrix[current_row][k] =   y_to_compare - current_y


			current_row+=1

			self.sliding_window_training_inputs = input_matrix
			self.sliding_window_training_outputs = output_matrix

		self.time_training_set = True
		
		return self.sliding_window_training_inputs,self.sliding_window_training_outputs 


	def create_training_set(self, time_type='sliding_window'):

		# Create time processed training sets if not done already

		if not self.time_training_set:

			if time_type=='sliding_window':

				self.sliding_window_training_set()

				self.current_training_input = self.sliding_window_training_inputs

			else:

				self.full_series_training_set()
				self.current_training_input = self.full_series_training_inputs


		# Pull out section from raw data matrix

		section = int(self.raw_data_matrix[0,2])

		section = [map_section(section)]

		one_hot_labels = keras.utils.to_categorical(section,num_classes=3)

		one_hot_labels = one_hot_labels[0] * np.ones((self.current_training_input.shape[0],1))

		#print(one_hot_labels)

		self.current_training_input = np.hstack((one_hot_labels,self.current_training_input))

		self.current_training_output = self.sliding_window_training_outputs






