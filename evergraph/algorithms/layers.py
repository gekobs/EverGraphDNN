import tensorflow as tf
from tensorflow import keras

class NtupleConv1D(tf.keras.layers.Layer):
    """

    """
    def __init__(self, n_tuple = 2, n_filters = 32, stride = 2, activation = "relu", name = "ntuple_conv_1d"):
        self.n_tuple = n_tuple
        self.n_filters = n_filters
        self.stride = stride
        self.activation = activation
        self.name = name

        self.num_outputs = 

    def build(self, input_shape):
        self.kernel = self.add_weight(
                "kernel",

