import pickle as cPickle
import tensorflow as tf
import model_audio_ANN
import numpy as np


pickle_path = './pickle_data/'
print("loading pickle files for ANN")
with open(pickle_path+"train_data_22k_org.pickle", "rb") as input_file:
    x_train = cPickle.load(input_file)

with open(pickle_path+"train_labels_22k_org.pickle", "rb") as input_file:
    y_train = cPickle.load(input_file)

with open(pickle_path+"test_data_22k_org.pickle", "rb") as input_file:
    x_test = cPickle.load(input_file)

with open(pickle_path+"test_labels_22k_org.pickle", "rb") as input_file:
    y_test = cPickle.load(input_file)

with open(pickle_path + 'valid_data_22k_org.pickle', 'rb') as input_file:
    x_valid = cPickle.load(input_file)

with open(pickle_path + "valid_labels_22k_org.pickle", "rb") as input_file:
    y_valid = cPickle.load(input_file)



x_train = np.row_stack([x_train, x_test])
y_train = np.row_stack([y_train, y_test])


ann_model_dir = './model/ANN/'

##############
# Train ANN ##
##############
NUM_EPOCHS = 500
BATCH_SIZE = 64
MODEL = model_audio_ANN.build_tflearn_ann(x_train.shape[1])
MODEL.fit(x_train, y_train, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(x_valid, y_valid),
          show_metric=True,
          batch_size=BATCH_SIZE)
MODEL.save(ann_model_dir+'Bee_audio_ANN.tfl')

print(MODEL.evaluate(x_test, y_test))
print(MODEL.evaluate(x_train, y_train))



# tf.reset_default_graph()
# ann_model_dir = './model/ANN_new/Bee_audio_ANN.tfl'
# ann_model = model_audio_ANN.build_tflearn_ann(x_test.shape[1])
# ann_model.load(ann_model_dir, weights_only=True, create_new_session = False)
# print(ann_model.evaluate(x_test, y_test))
# print(ann_model.evaluate(x_train, y_train))
# validation_acc = ann_model.evaluate(x_test, y_test)
# print(validation_acc)
