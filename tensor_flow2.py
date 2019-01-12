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

# create a path to directory where we have data
DATADIR = "D:\Projects\Python\SDDeepLearning_Datasets\kagglecatsanddogs_3367a\PetImages"
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
