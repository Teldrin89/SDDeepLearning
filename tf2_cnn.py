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

