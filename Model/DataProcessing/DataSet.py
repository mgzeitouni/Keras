import csv
import numpy as np
from .algebra_functions import *
import pdb


class DataSet:

	def __init__(self, path, header):

		self.processed = False

		self.processed_matrix = None

		self.training_inputs = None

		self.training_outputs = None

		with open(path, 'rU') as data_file:

			reader = csv.reader(data_file)

			if header:
				next(reader,None)

			data = np.array([row for row in reader])


		self.raw_data_matrix = data


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

			y = row[1]

			new_time = int((float(time)*processed_length)/raw_length)
			
			index = 0

			# Find and replace in the new matrix
			for row in self.new_unprocessed_matrix:

				if new_time==row[0]:
					
					self.new_unprocessed_matrix[index,1] = y

				index+=1

		if extrapolate_method=='connect_points':

			self.processed_time_series = connect_points(self.new_unprocessed_matrix)
			self.processed = True



	def training_set(self, output_type='regular'):

		# If time series not yet processed - then process

		if self.processed == False:
			self.process_data()


		# Create Input Matrix for Training Set

		n_data_points = self.processed_time_series.shape[0]

		time_array = self.processed_time_series[:,0]
		y_array = self.processed_time_series[:,1]

		self.training_outputs = np.array([y_array for i in range(len(y_array))])

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


		self.training_inputs = input_matrix

		if output_type=="zeros":

			self.training_outputs = output_matrix


		return self.training_inputs,self.training_outputs 






