# Audio-Classification-using-CNN-MLP
## Multi class audio classification using Deep Learning (CNN, MLP)

### Project Objectives: 
The objective of this project is to build a multi class classifier to identify sound
of a bee, cricket or noise.

### Dataset Description:
Given dataset contains total of 9,914 audio sample, where 3,300 belongs to Bee, 3,500 belongs to Cricket and 3,114 belongs to noise. Each audio sample is approximately about 2 sec long and has 44,100 amplitude samples/sec. Given dataset was merged and experiments were performed on 80%-20% split.

|            | Bee           | Cricket        |  Noise   | Total | 
|-----| ------------- |:-------------:| -----:| -----:|
|Train   | 2,402      | 3,000        | 2,180 | 7,582 |
|Test    | 898        | 500      | 934 |  2,332|
|        | 3,300      | 3,500     |    3,114 | 9,914 |


### Audio Data Preprocessing:

Audio dataset given has very high frame rate, on an average every file had 80,000 frames (amplitude/sec). With frames/sec being so high we have a lot of data and it needs some preprocessing. Reduction of audio frame rate and length was performed using [interpolation technique](https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html). The audio sample was reduced to 15k sample and total length of 22,000 (approximately 1/4 reduction of the given audio).
![alt text](https://raw.githubusercontent.com/vishalshar/Audio-Classification-using-CNN-MLP/master/img/audio_preprocessing-1.png)


### Using CNN

A network using Convolution layers was used to build classifier, network architecture is shown in Fig 6. The number of filters for both convolution was 64 and filter_size was 10 and 3 for respective layers followed by 3 fully connected layers, details about activation function used is in code. Max pooling was used after each convolution layer. During training over fitting was observed, to handle that dropout of 50% (keep) was used after first two fully connected layers and also ‘L2’ regularization was added to both layers. Input length was fixed as 22,000 with 1 channel. During training it was also observed, without downsampling
data model was not able to generalize well between bee and noise data. Adding downsampling technique helped the
model in generalization.

## Performance: 

Training was done for 500 epochs using Adaptive Moment Estimation (adam) as optimizer with learning rate of 0.0001. 


|            | Training           | Testing        |
|-----| ------------- |:-------------:|
|Accuracy   | 99.88%      | 99.45%       |



![alt text](https://raw.githubusercontent.com/vishalshar/Audio-Classification-using-CNN-MLP/master/img/ANN_Net_2-1.png)
![alt text](https://raw.githubusercontent.com/vishalshar/Audio-Classification-using-CNN-MLP/master/img/CNN_Net-1.png)
![alt text](https://raw.githubusercontent.com/vishalshar/Audio-Classification-using-CNN-MLP/master/img/audio_graph-1.png)
![alt text](https://raw.githubusercontent.com/vishalshar/Audio-Classification-using-CNN-MLP/master/img/bee_ann_audio.png)
![alt text](https://raw.githubusercontent.com/vishalshar/Audio-Classification-using-CNN-MLP/master/img/bee_cnn_audio.png)
