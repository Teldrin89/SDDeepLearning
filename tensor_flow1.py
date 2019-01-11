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
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
