import model_audio_ANN
import pickle as cPickle
import tensorflow as tf
import tflearn.helpers.evaluator


print("Load evaluation dataset")
pickle_path = './pickle_data/'


# f = open(pickle_path+'train_labels_22k_org.pickle', 'wb')
# f = open(pickle_path+'valid_labels_22k_org.pickle', 'wb')
# with open(pickle_path+"test_data_22k_org.pickle", "rb") as input_file:
with open(pickle_path + 'valid_data_22k_org.pickle', 'rb') as input_file:
    x_test = cPickle.load(input_file)

# with open(pickle_path+"test_labels_22k_org.pickle", "rb") as input_file:
with open(pickle_path + "valid_labels_22k_org.pickle", "rb") as input_file:
    y_test = cPickle.load(input_file)


tf.reset_default_graph()

# ann_model_dir = './model/ANN/'
ann_model_dir = './audio_model/ANN/'
# ann_model_dir = './model/ANN_new/Bee_audio_ANN.tfl'

ann_model = model_audio_ANN.build_tflearn_ann(x_test.shape[1])
ann_model.load(ann_model_dir+'Bee_audio_ANN.tfl', weights_only=True, create_new_session = False)
validation_acc = ann_model.evaluate(x_test, y_test, batch_size=64)
print(validation_acc)

# Best Acc: 0.9365351630524951