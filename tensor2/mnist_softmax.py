# Modified National Institute of Standards and Technology database
# is a large database of handwritten digits that is commonly 
# used for training various image processing systems.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# This is a classic case where a softmax regression is a natural, simple model. 
# If you want to assign probabilities to an object being one of several different things, 
# softmax is the thing to do, because
# softmax gives us a list of values between 0 and 1 that add up to 1. 
# Even later on, when we train more sophisticated models,
# the final step will be a layer of softmax.
