import numpy as np
import tensorflow as tf
import tflearn
import tflearn.layers.merge_ops
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



# length = 20000
drop_out_prob = 0.5
def build_tflearn_ann(length):
    input_layer = input_data(shape=[None, length, 1])


    pool_layer_1 = max_pool_1d(input_layer, 10, name='pool_layer_1')
    pool_layer_2 = max_pool_1d(pool_layer_1, 5, name='pool_layer_2')
    pool_layer_3 = max_pool_1d(pool_layer_2, 5, name='pool_layer_3')
    pool_layer_4 = max_pool_1d(pool_layer_3, 5, name='pool_layer_3')

    fully_connect_1 = fully_connected(pool_layer_3, 512, activation='relu', name='fully_connect_1',
                                      weights_init='xavier', regularizer="L2")

    fully_connect_2 = fully_connected(pool_layer_2, 512, activation='relu', name='fully_connect_2',
                                      weights_init='xavier', regularizer="L2")

    fully_connect_3 = fully_connected(pool_layer_1, 512, activation='relu', name='fully_connect_3',
                                      weights_init='xavier', regularizer="L2")

    fully_connect_4 = fully_connected(pool_layer_4, 512, activation='relu', name='fully_connect_3',
                                      weights_init='xavier', regularizer="L2")
    # Merge above layers
    merge_layer = tflearn.merge_outputs([fully_connect_1, fully_connect_2, fully_connect_3, fully_connect_4])
    # merge_layer = tflearn.merge_outputs(
    #     [fully_connect_1, fully_connect_2, fully_connect_3, fully_connect_4, fully_connect_5])
    # merge_layer = tflearn.merge_outputs(
    #     [fully_connect_1, fully_connect_2, fully_connect_3, fully_connect_4, fully_connect_5, fully_connect_6,
    #      fully_connect_7, fully_connect_8, fully_connect_9, fully_connect_10])
    drop_2 = dropout(merge_layer, 0.25)


    fc_layer_4 = fully_connected(drop_2, 2048, activation='relu', name='fc_layer_4', regularizer='L2',
                                 weights_init='xavier', weight_decay=0.001)
    drop_2 = dropout(fc_layer_4, drop_out_prob)


    fc_layer_5 = fully_connected(drop_2, 1024, activation='relu', name='fc_layer_5', regularizer='L2',
                                 weights_init='xavier', weight_decay=0.001)
    drop_3 = dropout(fc_layer_5, drop_out_prob)

    fc_layer_6 = fully_connected(drop_3, 128, activation='relu', name='fc_layer_5', regularizer='L2',
                                 weights_init='xavier', weight_decay=0.001)
    drop_4 = dropout(fc_layer_6, drop_out_prob)

    # Output
    fc_layer_2 = fully_connected(drop_4, 3, activation='softmax', name='output')
    network = regression(fc_layer_2, optimizer='adam', loss='softmax_categorical_crossentropy', learning_rate=0.0001,
                         metric='Accuracy')
    model = tflearn.DNN(network)
    return model