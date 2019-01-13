# introduction to deep learning - part2
# the second machine learning model will deal with identification if on given photo there is a dog or a cat
# for that task we take dataset from microsoft (used for kaggle competition in 2017)
# dataset has been put in a separate directory
# for the data preparation and model building we need numpy, matplotlib, cv2 and os libraries
# matplotlib will be needed to show some of the images, os will be needed to iterate in directory, cv2 for image
# operations and numpy for array operations
# numpy
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

# create a path to directory where we have data
DATADIR = "D:\Projects\Python\SDDeepLearning_Datasets"
# set categories
CATEGORIES = ["Dog", "Cat"]
# iterate through all examples of dog and cat
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        # loop through all images, changed to grey scale (without this the data would be much larger, sort of RBG)
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        # show 1st image in grays scale - commented once checked
        # plt.imshow(img_array, cmap="gray")
        # plt.show()
        # show image size - entry step to normalize pictures
        print(img_array.shape)
        # to get each image as a single shape use cv2 resize function
        # set image size
        IMG_SIZE = 50
        # creating new array with re-sized, normalized images
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        # show normalized picture to decide on size - commented once checked
        plt.imshow(new_array, cmap= "gray")
        plt.show()
        break
    break

# create a training data list
training_data = []


# create a function for creation of training data - taken most of the code from previous part of
# checking the normalization
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        # the neural network need to deal with numbers so the "Dog" will be 0 and "Cat" will be 1 (taken as indexes
        # from CATEGORIES list)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


# run the function to create training data
create_training_data()
# printout the length of training data
print(len(training_data))
# the results printout that there are 57 images that are corrupted but it still end without error and saves
# over 24900 images
# since the images area read from 2 different directories it would have all half cats and next one dogs (bad for model
# to learn it) so we have to shuffle data
random.shuffle(training_data)
# check if the shuffle worked
for sample in training_data[:10]:
    print(sample[1])
    break
# shuffled data will be packed inside the variables: x (input data - feature set) and y (output - categories)
# create empty lists
X = []
y = []
# run a for for loop to fill out X and y
for features, labels in training_data:
    X.append(features)
    y.append(labels)

# for neural network the X has to be converted from list to numpy array (first) and also reshape to be
# "-1"- how many features (in this case "-1" means no specific value), then image size (2 times as this is a 2
# dimensional array) and then 1 (as it is gray scale only, in case of for example different colors we'd have 3 -
# for RGB color scale)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# to save dataset - reshaped array for X and list for y - we can use pickle library
# create a pickle file
pickle_out = open("X.pickle", "wb")
# save dataset to opened pickle file
pickle.dump(X, pickle_out)
# close pickle file
pickle_out.close()
# the same script for y dataset
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# to open saved pickle
pickle_in = open("X.pickle", "rb")
# load data to X array
X = pickle.load(pickle_in)
# check entry 1
print(X[1])
