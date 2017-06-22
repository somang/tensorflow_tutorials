import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)