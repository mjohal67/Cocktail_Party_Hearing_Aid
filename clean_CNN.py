import tensorflow as tf
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Reshape
import tensorflow.keras.losses
from tensorflow.keras.callbacks import TensorBoard
import time


X = np.load("X_good.npy")
y = np.load("y_good.npy")

#would we have to flatten our data beforehand like they did in Matlab?

n_fft = 512
numNodes = (1 + n_fft//2) * 251 #double check this second dimension here - it's the number of time bins

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=X.shape[1:]))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#experement with adding more convolution layers (maybe last one without max pooling? --> find out what this would do)

model.add(Flatten()) #the shape would now be (1, numNodes)

model.add(Dropout(0.2)) #check to see if this dropout makes a difference in terms of accuracy (the other one we should keep, just check this one)
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(numNodes)) #outputting our prediction as the total flattened shape of our stft
model.add(Activation('relu')) #look at different types of activation functions to use here

model.add(Reshape((257, 251))) #I think this should work, but just in case, try reshaping data BEFOREHAND (in prepro file) into (-1, numNodes, 1), and then using 1D convolutions and pooling

model.summary() #summarizes all the layers and parts of our model


model.compile(loss='mean_squared_error', optimizer="adam", metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1) #testing on 10% of our data

#CHECK --> batch size, epochs
