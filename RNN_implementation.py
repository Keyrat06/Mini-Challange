import tflearn
import pickle
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
tf.python.control_flow_ops = tf


# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

trainData = np.load('trainData.npz')
X = trainData['arr_0']
X = X.astype('float16')
Y = trainData['arr_1']
Y = tflearn.data_utils.to_categorical(Y, 100)


# Load validation data
validData = np.load('validData.npz')
validX = validData['arr_0']
validX = validX.astype('float16')
validY = validData['arr_1']
validY = tflearn.data_utils.to_categorical(validY, 100)

print("HERE")

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([128, 128], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 128, 128, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 100, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=200, validation_set=(validX, validY),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnet')