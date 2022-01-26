import tensorflow as tf
from tensorflow import keras

def selective_mse(dummy_value):
    def calc_selective_mse(y_true, y_pred):
        sq_err = tf.square(y_true - y_pred)
        sq_err_filtered = tf.where(
                y_true == dummy_value,
                tf.zeros_like(y_true),
                sq_err
        )
        return tf.reduce_mean(sq_err_filtered)    
    return calc_selective_mse

