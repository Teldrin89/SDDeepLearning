# file prepared for validation of the model for categorization
# of dogs and cats (checked with jupyter notebook - requires additional settings)
# the part of the validation will look the same as it has to prepare the test image
# in the same way
# import tensorflow and cv2 libraries
import cv2
import tensorflow as tf

# put the same categories as in training dataset - keep the same order!
CATEGORIES = ["Dog", "Cat"]


# define function for filepath of image and pre-processing
def prepare(filepath):
    # use the same size as in training script
    IMG_SIZE = 50
    # load the image as array with cv2 as gray scale
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # resize the array to the image size
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # return the proper array (image) with reshape
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# load saved model
model = tf.keras.models.load_model("64x3-CNN.model")
# run the prediction for selected image - folder inside the project dir
# predict function always takes a list!
prediction = model.predict([prepare("SDDeepLearning_Datasets_TEST/10.jpg")])
# printout the prediction
print(CATEGORIES[int(prediction[0][0])])
