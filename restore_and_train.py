import tensorflow as tf
import numpy as np
import math
import os
#import matplotlib.pyplot as plt
import random
from tqdm import tqdm
np.random.seed(0)

# Load data
chkpt_name = "conv2a.ckpt" # DEFINE THE NAME OF SAVED FILE TO BE READ HERE
trainData = np.load('trainData.npz')
train = trainData['arr_0']
trainlabels = trainData['arr_1']
validData = np.load('validData.npz')
valid = validData['arr_0']
validlabels = validData['arr_1']
testData = np.load('testData.npz')
test = testData['arr_0']
print("IMAGES LOADED")

def to_onehot(labels,nclasses=100):
    outlabels = np.zeros((len(labels),nclasses))
    for i,l in enumerate(labels):
        outlabels[i,l] = 1
    return outlabels

trainOneHot = to_onehot(trainlabels,100)
validOneHot = to_onehot(validlabels,100)

# These will be inputs
## Input pixels, image with one channel (gray)
x = tf.placeholder("float", [None, 128, 128, 3])
# Note that -1 is for reshaping
x_im = tf.reshape(x, [-1,128,128,3])
## Known labels
# None works during variable creation to be
# unspecified size
y_ = tf.placeholder("float", [None,100])

# 128 X 128 X 3
# Conv layer 1
num_filters1 = 33
winx1 = 3
winy1 = 3
W1 = tf.Variable(tf.truncated_normal(
    [winx1, winy1, 3 , num_filters1],
    stddev=1./math.sqrt(winx1*winy1)))
b1 = tf.Variable(tf.constant(0.1,
                shape=[num_filters1]))
xw = tf.nn.conv2d(x_im, W1,
                  strides=[1, 1, 1, 1],
                  padding='SAME')
h1 = tf.nn.relu(xw + b1)

# 128 X 128 X 33
num_filters2 = 33
winx2 = 3
winy2 = 3
W2 = tf.Variable(tf.truncated_normal(
    [winx2, winy2, num_filters1, num_filters2],
    stddev=1./math.sqrt(winx2*winy2)))
b2 = tf.Variable(tf.constant(0.1,
                shape=[num_filters2]))
xw2 = tf.nn.conv2d(h1, W2,
                  strides=[1, 1, 1, 1],
                  padding='VALID')
h2 = tf.nn.relu(xw2 + b2)

# 126 X 126 X 33
num_filters3 = 64
winx3 = 3
winy3 = 3
W3 = tf.Variable(tf.truncated_normal(
    [winx3, winy3, num_filters2, num_filters3],
    stddev=1./math.sqrt(winx3*winy3)))
b3 = tf.Variable(tf.constant(0.1,
                shape=[num_filters3]))
xw3 = tf.nn.conv2d(h2, W3,
                  strides=[1, 1, 1, 1],
                  padding='SAME')
h3 = tf.nn.relu(xw3 + b3)


# 126 X 126 X 64
#3x3 Max pooling, no padding on edges
p1 = tf.nn.max_pool(h3, ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1], padding='VALID')
                 

# 62 X 62 X 64
num_filters4 = 80
winx4 = 1
winy4 = 1
W4 = tf.Variable(tf.truncated_normal(
    [winx2, winy2, num_filters3, num_filters4],
    stddev=1./math.sqrt(winx4*winy4)))
b4 = tf.Variable(tf.constant(0.1,
     shape=[num_filters4]))
p1w4 = tf.nn.conv2d(p1, W4,
       strides=[1, 1, 1, 1], padding='SAME')
h4 = tf.nn.relu(p1w4 + b4)


# 62 X 62 X 80
num_filters5 = 192
winx5 = 3
winy5 = 3
W5 = tf.Variable(tf.truncated_normal(
    [winx5, winy5, num_filters4, num_filters5],
    stddev=1./math.sqrt(winx5*winy5)))
b5 = tf.Variable(tf.constant(0.1,
     shape=[num_filters5]))
p1w5 = tf.nn.conv2d(h4, W5,
       strides=[1, 1, 1, 1], padding='VALID')
h5 = tf.nn.relu(p1w5 + b5)

# 60 X 60 X 192               
# 2x2 Max pooling, no padding on edges
p2 = tf.nn.max_pool(h5, ksize=[1, 2, 2, 1],
     strides=[1, 2, 2, 1], padding='VALID')

# Need to flatten convolutional output
p2_size = np.product(
        [s.value for s in p2.get_shape()[1:]])
p2f = tf.reshape(p2, [-1, p2_size ])

# Dense layer
num_hidden = 250
W6 = tf.Variable(tf.truncated_normal(
     [p2_size, num_hidden],
     stddev=2./math.sqrt(p2_size)))
b6 = tf.Variable(tf.constant(0.2,
     shape=[num_hidden]))
h6 = tf.nn.relu(tf.matmul(p2f,W6) + b6)

# Drop out training
keep_prob = tf.placeholder("float")
h6_drop = tf.nn.dropout(h6, keep_prob)

# Output Layer
W7 = tf.Variable(tf.truncated_normal(
     [num_hidden, 100],
     stddev=1./math.sqrt(num_hidden)))
b7 = tf.Variable(tf.constant(0.1,shape=[100]))


# Define model
y = tf.nn.softmax(tf.matmul(h6_drop,W7) + b7)

# Climb on cross-entropy
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        y + 1e-50, y_))

# How we train
train_step = tf.train.GradientDescentOptimizer(
             0.01).minimize(cross_entropy)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(y,1),
                              tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(
           correct_prediction, "float"))

# Restore session
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, chkpt_name)
    
    # Actually train
    epochs = 500 # Epoch here is defined to be 100k images
    trainSize = len(train)
    validSize = len(valid)
    batchSize = 50 # changed from 500 'cos GPU memory is limited
    numBatches = trainSize//batchSize
    train_acc = np.zeros(numBatches*epochs//500)
    test_acc = np.zeros(numBatches*epochs//500)
    print('Saving model every %d steps' %100000//4//batchSize)
    
    for i in tqdm(range(numBatches*epochs), ascii=True):
        batch = random.sample(xrange(trainSize),batchSize)
        trainBatch = train[batch]
        trainLabelBatch = trainOneHot[batch]
        
        # Record accuracy & save checkpoint every 1/4 of data
        if i % (100000//4//batchSize) == 0:            
            # Check accuracy on train set
            A = accuracy.eval(session=sess, feed_dict={x: train[batch],
                y_: trainOneHot[batch], keep_prob: 1.0})
            train_acc[i//500] = A
            print("Training acc: %.5f" %A)
            
            # And now the validation set
            batch2 = random.sample(xrange(validSize),batchSize)
            validBatch = valid[batch2]
            validLabelBatch = validOneHot[batch2]
            A = accuracy.eval(session=sess, feed_dict={x: valid[batch2],
                y_: validOneHot[batch2], keep_prob: 1.0})
            test_acc[i//500] = A
            print("Validation acc: %.5f" %A)
            
            # Save variables checkpoint
            print("Saving model checkpoint..")
            saver.save(sess, "conv2a_partial.ckpt")
        train_step.run(session=sess, feed_dict={x: train[batch], y_: trainOneHot[batch], keep_prob: .5})
        
        # Save checkpoint AND remove previous checkpoint to save space. Done on every epoch
        if i%(100000//batchSize)==0:
            saver.save(sess, "conv2a_"+str(i//(100000//batchSize))+".ckpt")
            if (i>=100000):
                os.remove("conv2a_"+str(i//(100000//batchSize) - 1)+".ckpt")
    
    # Save the weights
    saver.save(sess, "conv2a.ckpt")
    
