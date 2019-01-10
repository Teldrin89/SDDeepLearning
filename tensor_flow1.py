# introduction to deep learning - part1
# need tensorflow version 1.10 or greater
import tensorflow as tf
# import matplotlib for visualisation


print(tf.__version__)

mnist = tf.keras.datasets.mnist  # 28x28 images of hand written digits 0-9
