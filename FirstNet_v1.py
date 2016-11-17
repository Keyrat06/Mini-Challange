import tensorflow as tf
import numpy as np
import math
#import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
np.random.seed(0)

#------Functions T0 Simplify Life!-------------#
def conv2d(x, W, padding = 'SAME'):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Load data
'''
TO BE DONE: 
1. Store images as fp16 between 0.0 to 1.0 (saves processing time)
2. Try using elu instead of relu6 (claimed to be better)
3. Try using adaptive gradient optimizer 
'''
# Load train data
trainData = np.load('trainData.npz')
train = trainData['arr_0']
trainlabels = trainData['arr_1']
train = train.astype('float16')
train = train/255.0
"""
train = train/255.0 does the same thing as this loop

for i in range(0, 100000):
    train[i] = train[i]/255.0
"""
# Load validation data
validData = np.load('validData.npz')
valid = validData['arr_0']
validlabels = validData['arr_1']
valid = valid.astype('float16')
valid = valid/255.0
"""
for i in range(0, 10000):
    valid[i] = valid[i]/255.0
"""


# Load test set. We don't need it for now (save loading time)
#testData = np.load('testData.npz')
#test = testData['arr_0']
print("IMAGES LOADED")

def to_onehot(labels,nclasses=100):
    outlabels = np.zeros((len(labels),nclasses))
    for i,l in enumerate(labels):
        outlabels[i,l] = 1
    return outlabels

trainOneHot = to_onehot(trainlabels,100)
validOneHot = to_onehot(validlabels,100)

# ----------------- Model ---------------------#
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
W1 = weight_variable([winx1, winy1, 3, num_filters1])
b1 = bias_variable([num_filters1])
h1 = tf.nn.elu(conv2d(x_im, W1) + b1)
#W1 = tf.Variable(tf.truncated_normal(
#    [winx1, winy1, 3 , num_filters1],
#    stddev=0.2/math.sqrt(winx1*winy1)))
#b1 = tf.Variable(tf.constant(random.uniform(0.05,0.15),
#                shape=[num_filters1]))
#xw = tf.nn.conv2d(x_im, W1,
#                  strides=[1, 1, 1, 1],
#                  padding='SAME')
#h1 = tf.nn.elu(xw + b1)

# 128 X 128 X 33
num_filters2 = 33
winx2 = 3
winy2 = 3
W2 = weight_variable([winx2, winy2, num_filters1, num_filters2])
b2 = bias_variable([num_filters2])
h2 = tf.nn.elu(conv2d(h1, W2,'VALID') + b2)
#W2 = tf.Variable(tf.truncated_normal(
#    [winx2, winy2, num_filters1, num_filters2],
#    stddev=0.2/math.sqrt(winx2*winy2)))
#b2 = tf.Variable(tf.constant(0.1,
#                shape=[num_filters2]))
#xw2 = tf.nn.conv2d(h1, W2,
#                  strides=[1, 1, 1, 1],
#                  padding='VALID')
#h2 = tf.nn.elu(xw2 + b2)

# 126 X 126 X 33
num_filters3 = 64
winx3 = 3
winy3 = 3
W3 = weight_variable([winx3, winy3, num_filters2, num_filters3])
b3 = bias_variable([num_filters3])
h3 = tf.nn.elu(conv2d(h2, W3,'SAME') + b3)
#W3 = tf.Variable(tf.truncated_normal(
#    [winx3, winy3, num_filters2, num_filters3],
#    stddev=0.2/math.sqrt(winx3*winy3)))
#b3 = tf.Variable(tf.constant(random.uniform(0.15,0.25),
#                shape=[num_filters3]))
#xw3 = tf.nn.conv2d(h2, W3,
#                  strides=[1, 1, 1, 1],
#                  padding='SAME')
#h3 = tf.nn.elu(xw3 + b3)


# 126 X 126 X 64
#3x3 Max pooling, no padding on edges
p1 = tf.nn.max_pool(h3, ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1], padding='VALID')
                 

# 62 X 62 X 64
num_filters4 = 128
winx4 = 1
winy4 = 1
W4 = weight_variable([winx4, winy4, num_filters3, num_filters4])
b4 = bias_variable([num_filters4])
h4 = tf.nn.elu(conv2d(p1, W4, 'SAME') + b4)
#W4 = tf.Variable(tf.truncated_normal(
#    [winx4, winy4, num_filters3, num_filters4],
#    stddev=0.2/math.sqrt(winx4*winy4)))
#b4 = tf.Variable(tf.constant(random.uniform(0.05,0.35),
#     shape=[num_filters4]))
#p1w4 = tf.nn.conv2d(p1, W4,
#       strides=[1, 1, 1, 1], padding='SAME')
#h4 = tf.nn.elu(p1w4 + b4)

# 62 X 62 X 192 
# (CHIHEEM): swapped order cos it is seems weird to have a huge bank at the end
num_filters5 = 80
winx5 = 3
winy5 = 3
W5 = weight_variable([winx5, winy5, num_filters4, num_filters5])
b5 = bias_variable([num_filters5])
h5 = tf.nn.elu(conv2d(h4, W5, 'VALID') + b5)
#W5 = tf.Variable(tf.truncated_normal(
#    [winx5, winy5, num_filters4, num_filters5],
#    stddev=0.2/math.sqrt(winx5*winy5)))
#b5 = tf.Variable(tf.constant(random.uniform(0.05,0.15),
#     shape=[num_filters5]))
#p1w5 = tf.nn.conv2d(h4, W5,
#       strides=[1, 1, 1, 1], padding='VALID')
#h5 = tf.nn.elu(p1w5 + b5)

# 60 X 60 X 80
# 2x2 Max pooling, no padding on edges
p2 = tf.nn.max_pool(h5, ksize=[1, 2, 2, 1],
     strides=[1, 2, 2, 1], padding='VALID')

# Input is now about 30x30x192
# Need to flatten convolutional output
p2_size = np.product(
        [s.value for s in p2.get_shape()[1:]])
p2f = tf.reshape(p2, [-1, p2_size ])

# Dense layer
num_hidden_a = 256
W6a = tf.Variable(tf.truncated_normal([p2_size, num_hidden_a],stddev=0.1))
#W6a = tf.Variable(tf.truncated_normal(
#     [p2_size, num_hidden_a],
#     stddev=0.2/math.sqrt(p2_size)))
b6a = tf.Variable(tf.constant(random.uniform(0,1.0),
     shape=[num_hidden_a]))
h6a = tf.nn.elu(tf.matmul(p2f,W6a) + b6a)

# Drop out training
keep_prob = tf.placeholder("float")
h6_drop = tf.nn.dropout(h6a, keep_prob)

# Output Layer
W7 = tf.Variable(tf.truncated_normal([num_hidden_a, 100], stddev=0.1))
#W7 = tf.Variable(tf.truncated_normal(
#     [num_hidden_a, 100],
#     stddev=0.1/math.sqrt(num_hidden_a)))
b7 = tf.Variable(tf.constant(0.1,shape=[100]))
y_logit = tf.matmul(h6_drop,W7) + b7

#-------------------- End model specification ----------------#

# Climb on cross-entropy
# (CHIHEEM) - Changed because the old way is not right. See documentations
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    y_logit + 1e-50, #logits 
                                    y_       #actual class labels
                                ))
''' OLD CODES (should be wrong):
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        y+ 1e-50, y))
'''

# Setup up exponential decay
global_step = tf.Variable(0.0, trainable=False)
learning_rate = tf.train.exponential_decay(1e-1, global_step,
                                           100.0, 0.96, staircase=True)
#learning_rate = 1e-3 # Comment this line off if you don't want fixed rate

# activate this to use adaptive gradient
train_step = tf.train.AdagradOptimizer(
     learning_rate).minimize(cross_entropy, global_step=global_step)
'''
train_step = tf.train.GradientDescentOptimizer(
     learning_rate).minimize(cross_entropy, global_step=global_step)
'''
# Define accuracy
y_softmax = tf.nn.softmax(y_logit+1e-20) # the epilson is for safety
correct_prediction = tf.equal(tf.argmax(y_softmax,1),
                              tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(
           correct_prediction, "float"))

# Set up saver
saver = tf.train.Saver()

# Train!
with tf.Session() as sess:     
    # Initialize variables
    sess.run(tf.initialize_all_variables())
    
    epochs = 20 # Epoch here is defined to be 100k images
    trainSize = len(train)
    validSize = len(valid)
    batchSize = 100 # changed from 500 'cos GPU memory is limited
    numBatchesPerEpoch = trainSize//batchSize
    
    # Compute initial loss for reference
    y_softmax_initial = y_softmax.eval(feed_dict = {x:train[0:100000:1000], keep_prob: 1.0})
    loss_initial = tf.reduce_mean(-tf.reduce_sum(trainlabels[0:100000:1000]*tf.log(y_softmax_initial), 1)).eval()
    print("%6d. loss = %s" %(-1, loss_initial))
    
    save_per_steps = 100000//2//batchSize # Define how many steps to run before saving. Default is 50k images
    print('Saving model every %d steps' %save_per_steps)
    
    f = open('trainingStatus.txt', 'wb')
          
    for i in range(numBatchesPerEpoch*epochs): #tqdm(range(numBatchesPerEpoch*epochs), ascii=True):
        
        # Set up training batch
        batch = random.sample(range(trainSize),batchSize)
        trainBatch = train[batch]
        trainLabelBatch = trainOneHot[batch]
        
        # Run one iteration of training
        _, loss_val = sess.run([train_step, cross_entropy], 
                               feed_dict={x: trainBatch, 
                                          y_: trainLabelBatch, 
                                          keep_prob: 0.5
                                          })
        
        
        # Debugging lines
        if i%20==0:
            y_softmax_initial = y_softmax.eval(feed_dict = {x:trainBatch, keep_prob: 1.0})
            loss_initial = tf.reduce_mean(-tf.reduce_sum(trainLabelBatch*tf.log(y_softmax_initial), 1)).eval()
            print("%6d. loss = %s" %(-1, loss_initial))
            print("%6d. loss = %s (Learning rate: %.8f)" %(i, loss_val, learning_rate.eval()))
            print(W1.eval()[0][0][0][0], W2.eval()[1][1][1][0], W3.eval()[2][2][2][0])
            #print(tf.argmax(y_softmax.eval(feed_dict = {x:trainBatch, keep_prob: 1.0}),1).eval())
            #print(tf.argmax(y_.eval(feed_dict={y_: trainLabelBatch}),1).eval())
            #train_acc = accuracy.eval(session=sess, feed_dict={x: trainBatch,
            #    y_: trainLabelBatch, keep_prob: 1.0})
            #print("Training acc: %.5f" %train_acc)
        
            
        # Record accuracy & save checkpoint
        if (i % save_per_steps== 0) & (i>0) :            
            # Check accuracy on train set
            train_acc = accuracy.eval(session=sess, feed_dict={x: trainBatch,
                y_: trainLabelBatch, keep_prob: 1.0})
            print("Training acc: %.5f" %train_acc)
            
            # And now the validation set
            
            batchesForValidation = validSize//batchSize
            totalAcc = 0
            for j in range(0,batchesForValidation):
                validation_sub_acc = accuracy.eval(session=sess,
                                                   feed_dict={x: valid[j*batchSize:(j+1)*batchSize-1],
                                                              y_: validOneHot[j*batchSize:(j+1)*batchSize-1],
                                                              keep_prob: 1.0})
                totalAcc += validation_sub_acc*batchSize
            validation_acc = totalAcc/validSize
            print("Validation acc: %.5f" %validation_acc)
            
            f.write("%5.2f, %.6f \n" %(i/numBatchesPerEpoch, validation_acc))
            
            # Save variables checkpoint
            print("Saving model checkpoint..")
            saver.save(sess, "conv2a_partial.ckpt")
                    
        # Save checkpoint AND remove previous checkpoint to save space (~150MB per file). 
        # Done on every epoch
        if i%(100000//batchSize)==0 & i>0:
            saver.save(sess, "conv_"+str(i//(100000//batchSize))+".ckpt")
            if (i>=(100000//batchSize)):
                try:
                    os.remove("conv_"+str(i//(100000//batchSize) - 1)+".ckpt")
                except:
                    pass
    
    # Save the weights after all the training has been done
    saver.save(sess, "conv_final.ckpt")




"""
### EXPEREMENTING HERE###

# These will be inputs
## Input pixels, flattened
x = tf.placeholder("float", [None, 128*128*3])
## Known labels
y_ = tf.placeholder("float", [None,100])

# Variables
W = tf.Variable(tf.zeros([128*128*3,100]))
b = tf.Variable(tf.zeros([100]))

# Just initialize
sess.run(tf.initialize_all_variables())

# Define model
y = tf.nn.softmax(tf.matmul(x,W) + b)

### End model specification, begin training code


# Climb on cross-entropy
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        y + 1e-50, y_))

# How we train
train_step = tf.train.GradientDescentOptimizer(
                0.1).minimize(cross_entropy)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(y,1),
                     tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(
           correct_prediction, "float"))


# Actually train
batchSize = 200
epochs = 1000
rangeSize = len(train)
validSize = len(valid)
train_acc = np.zeros(epochs//10)
test_acc = np.zeros(epochs//10)
for i in tqdm(range(epochs)):
    batch = random.sample(range(rangeSize),batchSize)
    trainBatch = train[batch]
    trainLabelBatch = trainOneHot[batch]
    # Record summary data, and the accuracy
    if i % 10 == 0:
        batch2 = random.sample(range(validSize),batchSize)
        validBatch = valid[batch2]
        validLabelBatch = validOneHot[batch2]
        # Check accuracy on train set
        A = accuracy.eval(feed_dict={
            x: trainBatch.reshape([-1,128*128*3]),
            y_: trainLabelBatch})
        train_acc[i//10] = A
        # And now the validation set
        A = accuracy.eval(feed_dict={
            x: validBatch.reshape([-1,128*128*3]),
            y_: validLabelBatch})
        test_acc[i//10] = A
    train_step.run(feed_dict={
        x: trainBatch.reshape([-1,128*128*3]),
        y_: trainLabelBatch})
"""
                   
