from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.utils
import numpy as np

# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(500, activation='relu', input_dim=100))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='adam',
              loss='mean_squared_error')

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))


# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)