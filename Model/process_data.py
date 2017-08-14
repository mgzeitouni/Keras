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

	input_training_file_created = False

	for subdir, dirs, files in os.walk('training_sets/inputs'):

		for file in files:

			if '.DS_Store' not in file:

				with open('training_sets/inputs/%s' %file, 'rU') as data_file:

					reader = csv.reader(data_file)

					data = np.array([row for row in reader])

					write_type = 'a'

					if not input_training_file_created:

						write_type='w+'


					with open('training_sets/all/inputs/%s.csv' %timestamp, write_type) as all_data_file:

						writer = csv.writer(all_data_file)

						input_training_file_created = True

						writer.writerows(data)

	output_training_file_created = False


	for subdir, dirs, files in os.walk('training_sets/outputs'):

		for file in files:

			if '.DS_Store' not in file:

				with open('training_sets/outputs/%s' %file, 'rU') as data_file:

					reader = csv.reader(data_file)

					data = np.array([row for row in reader])

					write_type = 'a'

					if not output_training_file_created:

						write_type='w+'

					with open('training_sets/all/outputs/%s.csv' %timestamp, write_type) as all_data_file:

						writer = csv.writer(all_data_file)

						output_training_file_created = True

						writer.writerows(data)



	end = time.time()

	print ("Consolidation Time: %0.2f minutes" %((end-start)/60))

if __name__=='__main__':


	#create_training_files('../sample_data')
	
	consolidate_training_files()
