# to check if the tensorflow model is saved in separate file we try to run it for
# predictions and check a single case
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# To disable the the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# use a template dataset from keras api - it's kind of "hello world" example for ML
mnist = tf.keras.datasets.mnist  # 28x28 images of hand written digits 0-9
# load dataset to test and train sets (use only test set!)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# load saved model
saved_model = tf.keras.models.load_model("simple_num_reader.model")
# using the test dataset we use the new_model (loaded one) and run for all of the x_test data (inputs)
test_predictions = saved_model.predict([x_test])
# use numpy to get a single case from predictions and run it against the picture itself (using matplotlib)
# show image
plt.imshow(x_test[15])
plt.show()
# check prediction
print(np.argmax(test_predictions[15]))
