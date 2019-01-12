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
model.add(tf.keras.layers.Flatten())
# add a 2nd layer (hidden) - first one with neurons - it will be a dense layer with set number of units (128 neurons)
# and activation function set up to rectified linear function (default option)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# add a 3rd layer - second hidden layer, the same as previous one
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# add last layer - the output one - it will have 10 output possibilities (digits 0-9) and the activation function
# will be a shape of probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# todo: finished @12:43
