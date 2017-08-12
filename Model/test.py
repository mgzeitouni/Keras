from DataProcessing import DataSet
import numpy as np
import matplotlib.pyplot as plt

path = '../sample_data/sample1.csv'

mydata = DataSet(path, header=True)

mydata.process_data(days_back=60, extrapolate_method='connect_points')

x,y = mydata.training_set(output_type='regular')

print(y)
#print(mydata.training_inputs[60:80, 20:60])

#print(mydata.new_unprocessed_matrix)
# x = mydata.processed_time_series[:,0]
# y = mydata.processed_time_series[:,1]
# #print (new_matrix[0:15,:])
# plt.plot(x,y)
# plt.show()