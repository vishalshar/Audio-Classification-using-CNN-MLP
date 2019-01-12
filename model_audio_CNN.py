import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, flatten, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.conv import conv_1d, max_pool_1d, avg_pool_1d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
import pickle as cPickle
import tflearn.datasets.mnist as mnist
import os
import cv2
import numpy as np
import scipy.io.wavfile
import random



drop_out_prob = 0.25
def build_tflearn_cnn(length):
    input_layer = input_data(shape=[None, length, 1])


    # Convolution Layer
    conv_layer_1 = conv_1d(input_layer,
                           nb_filter=512,
                           filter_size=10,
                           activation='relu',
                           name='conv_layer_1',
                           weights_init='xavier',
                           regularizer="L2")
    pool_layer_1 = max_pool_1d(conv_layer_1, 4, name='pool_layer_1')


    conv_layer_2 = conv_1d(pool_layer_1,
                           nb_filter=512,
                           filter_size=5,
                           activation='relu',
                           name='conv_layer_2',
                           weights_init='xavier',
                           regularizer="L2")
    pool_layer_3 = max_pool_1d(conv_layer_2, 4, name='pool_layer_3')
    # flat = flatten(pool_layer_3)


    fc_layer_4 = fully_connected(pool_layer_3, 256, activation='relu', name='fc_layer_4', regularizer='L2')
    drop_2 = dropout(fc_layer_4, drop_out_prob)
    fc_layer_5 = fully_connected(drop_2, 128, activation='relu', name='fc_layer_5', regularizer='L2')
    drop_3 = dropout(fc_layer_5, drop_out_prob)

    # Output
    fc_layer_2 = fully_connected(drop_3, 3, activation='softmax', name='output')
    network = regression(fc_layer_2, optimizer='adam', loss='softmax_categorical_crossentropy', learning_rate=0.0001,
                         metric='accuracy')
    model = tflearn.DNN(network,
                        tensorboard_verbose=0)

    return model



