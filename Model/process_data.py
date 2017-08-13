import csv
import numpy as np
import pdb
from DataProcessing import DataSet
import os
import keras.utils
import time

def create_training_files(data_dir):

	for subdir, dirs, files in os.walk(data_dir):

		for file in files:

			sample_name = file.replace('.csv', '')

			# Load this file's data

			if '.DS_Store' not in file:

				path = "%s/%s" %(data_dir,file)

				print(path)

				with open(path, 'rU') as data_file:

					reader = csv.reader(data_file)

					next(reader,None)

					data = np.array([row for row in reader])


				processed_data = DataSet(path, data)

				processed_data.create_training_set()

				# Write to input file
				
				with open("training_sets/inputs/%s_training_inputs.csv" %sample_name, 'w+') as training_file:

					writer = csv.writer(training_file)

					writer.writerows(processed_data.current_training_input)

				
				with open("training_sets/outputs/%s_training_outputs.csv" %sample_name, 'w+') as training_file:

					writer = csv.writer(training_file)

					writer.writerows(processed_data.current_training_output )
				



def consolidate_training_files():

	start = time.time()

	timestamp = str(time.time())[0:10]

	all_inputs_matrix_created = False
	all_outputs_matrix_created = False

	for subdir, dirs, files in os.walk('training_sets/inputs'):

		for file in files:

			if '.DS_Store' not in file:

				with open('training_sets/inputs/%s' %file, 'rU') as data_file:

					reader = csv.reader(data_file)

					data = np.array([row for row in reader])

					if not all_inputs_matrix_created:
						all_inputs_matrix = data
						all_inputs_matrix_created = True

					else:
						all_inputs_matrix = np.vstack((all_inputs_matrix,data))

	with open('training_sets/all/inputs/%s.csv' %timestamp, 'w+') as new_file:

		writer = csv.writer(new_file)
		writer.writerows(all_inputs_matrix)

	for subdir, dirs, files in os.walk('training_sets/outputs'):

		for file in files:

			if '.DS_Store' not in file:

				with open('training_sets/inputs/%s' %file, 'rU') as data_file:

					reader = csv.reader(data_file)

					data = np.array([row for row in reader])

					if not all_outputs_matrix_created:
						all_outputs_matrix = data
						all_outputs_matrix_created = True

					else:
						all_outputs_matrix = np.vstack((all_outputs_matrix,data))

	with open('training_sets/all/outputs/%s.csv' %timestamp, 'w+') as new_file:

		writer = csv.writer(new_file)
		writer.writerows(all_outputs_matrix)

	end = time.time()

	print ("Consolidation Time: %0.2f minutes" %((end-start)/60))

if __name__=='__main__':


	#create_training_files('../sample_data')
	
	consolidate_training_files()
