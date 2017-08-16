from DataProcessing import DataSet
import numpy as np
import matplotlib.pyplot as plt
import csv

path = '../sample_data/sample1.csv'

with open(path, 'rU') as data_file:

	reader = csv.reader(data_file)

	next(reader,None)

	data = np.array([row for row in reader])


mydata = DataSet(path, data, window_width=0.5)

#mydata.moving_average()
print (mydata.moving_average_series)

#mydata.process_time_data(days_back=60, extrapolate_method='connect_points')

#print(mydata.processed_time_series[8500:mydata.processed_time_series.shape[0],:])

#mydata.create_training_set()

#x,y = mydata.training_set(output_type='regular')

#print(y)
#print(mydata.training_inputs[60:80, 20:60])

#print(mydata.new_unprocessed_matrix)
# x = mydata.processed_time_series[:,0]
# y = mydata.processed_time_series[:,1]
# #print (new_matrix[0:15,:])
# plt.plot(x,y)
# plt.show()