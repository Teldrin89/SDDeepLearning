# second file for part2 of tensor flow deep learning tutorial - in part1 there are instructions to build datasets
# for the input and output of training data of pictures of cats and dogs
# this file will contain the model and scripts for machine learning of created model
# import required libraries: tensorflow
import tensorflow as tf
# import from keras the model
from tensorflow.keras.models import Sequential
# import from keras the layers used - Dense and Flatten (for last layer)
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
# import pickle to load saved datasets
import pickle
# open input (X) and output (y) train datasets
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
# the next step is to normalize data - scale it - as the image data has max value of 255 each will be divide by max
X = X/255
# start building a model - sequential model
model = Sequential()
# add 1st layer - convolutional 2D layer, with 64 as filter, 3x3 window (snap) and input shape taken from X but
# without first (0) value as it stores the number of features instead of feature that we want
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
# still within 1st layer add activation layer (could be activation or pooling but in this example it will be
# activation -> pooling)
model.add(Activation("relu"))
# last part for 1st layer is max-pooling layer with pool size of 2x2
model.add(MaxPooling2D(pool_size=(2,2)))
