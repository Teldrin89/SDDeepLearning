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
# import tensor board - to save log from training
from tensorflow.keras.callbacks import TensorBoard
# import time library
import time

# it's a good practice to save each model separately as each time we overwrite the model it is not really overwriting
# but appending
# put a name of model - use time as additional portion (this way each one is unique)
NAME = "cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))
# specify the callback object
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# configuring the session to run tensorflow on gpu
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# open input (X) and output (y) train datasets
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
# the next step is to normalize data - scale it - as the image data has max value of 255 each will be divide by max
X = X/255
# start building a model - sequential model
model = Sequential()
# add 1st hidden layer - convolutional 2D layer, with 64 as filter, 3x3 window (snap) and input shape taken from X but
# without first (0) value as it stores the number of features instead of feature that we want
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
# still within 1st hidden layer add activation layer (could be activation or pooling but in this example it will be
# activation -> pooling)
model.add(Activation("relu"))
# last part for 1st hidden layer is max-pooling layer with pool size of 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# the second layer will look exactly the same as the previous one - this will change the ML model to deep learning
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# before passing the last layer, the data will be changed to dense layer - value set 64 as the filter from Conv2D
# data also has to "flattened" (as the one passing before is a 2D array) to 1D
model.add(Flatten())
# change data to dense layer with 64 nodes
model.add(Dense(64))
# add activation function
model.add(Activation("relu"))
# add output layer - also Dense, single node
model.add(Dense(1))
# add activation to last layer - sigmoid function
model.add(Activation("sigmoid"))
# compile model - selecting loss (binary as we have only 2 results possible), optimizer ("adam" as the standard one),
# and metrics (accuracy of the model)
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
# run the model fitting - training the actual model - use the X dataset input and y output, with batch size of 32
# (not to pass all the data at once), run it for 3 iterations and setting up the validation split to 10%
# add callbacks to model fitment - tensorboard
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])
model.save("{}.model".format(NAME))

# the next part is to work on the validation of the model training by using a tensor board - this will show how
# during iterations the model accuracy and loss (in and out of sample) change and based on that how to change the
# model to get better results

# to run tensorboard and see the results run this in cmd from the folder with model and logs
# "tensorboard --logdir=logs/" and then use the localhost it provides in chrome "localhost:00000"
