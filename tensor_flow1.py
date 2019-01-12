# introduction to deep learning - part1
# need tensorflow version 1.10 or greater
# keras - high level api of tensorflow
# the goal of machine learning model is to take the presented input data set and by the introduction of at least one
# hidden layer of neurons (neural network) or more than one (deep neural network) determine the output
# each neuron is taking a sum of inputs multiplied by each specific weight and then run against an activation function
# that will determine if the neuron should work or now (stepped function) although with more complex models the
# activation function takes shape of a sigmoid function

# import tensorflow library
import tensorflow as tf
# import matplotlib for visualisation
import matplotlib.pyplot as plt
# import numpy for the prediction check
import numpy as np

# printout simple function - get the tf version
print(tf.__version__)
# use a template dataset from keras api - it's kind of "hello world" example for ML
mnist = tf.keras.datasets.mnist  # 28x28 images of hand written digits 0-9
# the idea will be to train and test a model against this dataset so that it will determine a digit based on a picture
# load dataset to 2 sets of variables: train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# the data is given in a form of tensors - multidimensional array - here is an example of first entry
print(x_train[0])
# to see the actual picture - run the same entry against a imshow from pyplot library
plt.imshow(x_train[0], cmap=plt.cm.binary)  # use a binary color map for the picture
plt.show()  # show the picture
# once the dataset has been obtained it's good to normalize it (or at leas scale)
# the dataset loaded varies between 0 and 255 so to make them scale between 0 and 1 we use a keras normalize function
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# print again the image, see that it is slightly lighter
print(x_train[0])
plt.imshow(x_train[0], cmap=plt.cm.binary)  # use a binary color map for the picture
plt.show()  # show the picture
# the reason for normalization is that ML models work better with numbers below 1
# build the model - sequential type (most common, feed forward model)
model = tf.keras.models.Sequential()
# the data normalized is still a multidimensional array and in order to use a sequential model dat has to be flattened
# for other models (eg. convolutional) this wouldn't be needed
# to flatten data we can either use some additional functions from numpy or use a flatten function from keras
# add a input layer with flatten data
# UPDATE! In order to make the model.save function to work, the first layer of input data has to be described
# with given input shape (in our case a matrix - 28x28 in size)
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# add a 2nd layer (hidden) - first one with neurons - it will be a dense layer with set number of units (128 neurons)
# and activation function set up to rectified linear function (default option)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# add a 3rd layer - second hidden layer, the same as previous one
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# add last layer - the output one - it will have 10 output possibilities (digits 0-9) and the activation function
# will be a shape of probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# with the architecture of the model defined the next step is to define parameters for training of the module
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# the parameters to be defined for training are: optimizer - the standard go-to one is called "adam", loss - the
# categorical cross entropy is one of the standard method for calculation of loss, the metrics that will be tracked
# is just an accuracy
# the basic idea behind a machine learning model is that it aims not at getting the correct answer (value) but rather
# at minimizing loss (error)
# with the settings done for the compile of model we can run the model against train data
model.fit(x_train, y_train, epochs=3)
# the parameters used in fit function (the one that will train the model) are inputs, outputs (of train dataset) and
# number of iterations (epochs)

# to check the validation loss and accuracy use the tensorflow functions
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
# The results show a 96% accuracy and 0.11 loss - it's a high accuracy for just 3 epochs and it is also important
# to not let the model over-fit (it happens often that the model instead of learning patterns it simply memorizes the
# cases with which it was trained
# At this point the model is done and trained, to save a model use the save function
model.save("simple_num_reader.model")
# To load the same model use the load model function
new_model = tf.keras.models.load_model("simple_num_reader.model")
# Now, using the test dataset we use the new_model (loaded one) and run for all of the x_test data (inputs)
predictions = new_model.predict([x_test])
# Remember that the prediction function ALWAYS takes a list as input!
# printout the prediction results (as a multi-dimension array)
print(predictions)
# to see more user-friendly how single test case looks (if the prediction is matching result) use a numpy function
# and get the max argument for given test case
print(np.argmax(predictions[3]))
# check the actual test number from given case by plotting its image
plt.imshow(x_test[3])
plt.show()
