import model_audio_CNN
import pickle as cPickle
import tensorflow as tf
import tflearn.helpers.evaluator


print("Load evaluation dataset")
pickle_path = './pickle_data/'

with open(pickle_path + 'valid_data_22k_org.pickle', 'rb') as input_file:
# with open(pickle_path+"test_data_22k_org.pickle", "rb") as input_file:
    x_test = cPickle.load(input_file)

with open(pickle_path + "valid_labels_22k_org.pickle", "rb") as input_file:
# with open(pickle_path+"test_labels_22k_org.pickle", "rb") as input_file:
    y_test = cPickle.load(input_file)


tf.reset_default_graph()

cnn_model_dir = './audio_model/CNN/Bee_audio_CNN.tfl'
cnn_model = model_audio_CNN.build_tflearn_cnn(x_test.shape[1])
cnn_model.load(cnn_model_dir, weights_only=True)
validation_acc = cnn_model.evaluate(x_test, y_test, batch_size=32)
print(validation_acc)

# Best Acc: 0.9948542024013722