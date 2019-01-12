import model_audio_CNN
import pickle as cPickle
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# export CUDA_VISIBLE_DEVICE=0,1

pickle_path = './pickle_data/'
print("loading pickle files for CNN")
with open(pickle_path+"train_data_22k_org.pickle", "rb") as input_file:
    x_train = cPickle.load(input_file)

with open(pickle_path+"train_labels_22k_org.pickle", "rb") as input_file:
    y_train = cPickle.load(input_file)

with open(pickle_path+"test_data_22k_org.pickle", "rb") as input_file:
    x_test = cPickle.load(input_file)

with open(pickle_path+"test_labels_22k_org.pickle", "rb") as input_file:
    y_test = cPickle.load(input_file)


x_train = np.row_stack([x_train, x_test])
y_train = np.row_stack([y_train, y_test])


with open(pickle_path + 'valid_data_22k_org.pickle', 'rb') as input_file:
    x_valid = cPickle.load(input_file)

# with open(pickle_path+"test_labels_22k_org.pickle", "rb") as input_file:
with open(pickle_path + "valid_labels_22k_org.pickle", "rb") as input_file:
    y_valid = cPickle.load(input_file)


cnn_model_dir = './model/CNN/'

##############
# Train CNN ##
##############
NUM_EPOCHS = 500
BATCH_SIZE = 8
MODEL = model_audio_CNN.build_tflearn_cnn(x_train.shape[1])
# with tf.device('/gpu:0'):
MODEL.fit(x_train, y_train, n_epoch=NUM_EPOCHS,
              shuffle=True,
              validation_set=(x_valid, y_valid),
              show_metric=True,
              batch_size=BATCH_SIZE,
          run_id = 'Bee_audio_CNN_best_4')
MODEL.save(cnn_model_dir+'Bee_audio_CNN.tfl')





# ##################
# ## Evaluate CNN ##
# ##################
# tf.reset_default_graph()

# cnn_model_dir = '/home/vishal/PycharmProjects/bee_audio_project/model/CNN/Bee_audio_CNN_100.tfl'
# # cnn_model = model_audio_CNN.build_tflearn_cnn(x_test.shape[1])

# MODEL.load(cnn_model_dir, weights_only=True)
# validation_acc = MODEL.evaluate(x_test, y_test)
# print(validation_acc)