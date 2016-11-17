import tensorflow as tf
import numpy as np
import math
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
np.random.seed(0)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# The MNIST images are always 28x28 pixels.
numExamples = mnist[0]._num_examples
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

#------Functions T0 Simplify Life!-------------#
def conv2d(x, W, padding = 'SAME'):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# ----------------- Model ---------------------#
# These will be inputs
## Input pixels, image with one channel (gray)
x = tf.placeholder("float", [None, IMAGE_PIXELS])
# Note that -1 is for reshaping
x_im = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1])
## Known labels
# None works during variable creation to be
# unspecified size
y_ = tf.placeholder("float", [None,10])

# 28 X 28 X 1
# Conv layer 1x
num_filters1 = 33
winx1 = 3
winy1 = 3
W1 = weight_variable([winx1, winy1, 1, num_filters1])
b1 = bias_variable([num_filters1])
h1 = tf.nn.elu(conv2d(x_im, W1) + b1)

# 28 X 28 X 33
num_filters2 = 33
winx2 = 3
winy2 = 3
W2 = weight_variable([winx2, winy2, num_filters1, num_filters2])
b2 = bias_variable([num_filters2])
h2 = tf.nn.elu(conv2d(h1, W2,'VALID') + b2)

# 26 X 26 X 33
num_filters3 = 64
winx3 = 3
winy3 = 3
W3 = weight_variable([winx3, winy3, num_filters2, num_filters3])
b3 = bias_variable([num_filters3])
h3 = tf.nn.elu(conv2d(h2, W3,'SAME') + b3)


# 26 X 26 X 64
#3x3 Max pooling, no padding on edges
p1 = tf.nn.max_pool(h3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')


#I DONT TRUST THIS LAYER IN MY MIND THE MATH HERE IS WEIRD!
# 12 X 12 X 64
num_filters4 = 128
winx4 = 1
winy4 = 1
W4 = weight_variable([winx4, winy4, num_filters3, num_filters4])
b4 = bias_variable([num_filters4])
h4 = tf.nn.elu(conv2d(p1, W4, 'SAME') + b4)


# 12 X 12 X 80
# (CHIHEEM): swapped order cos it is seems weird to have a huge bank at the end
num_filters5 = 80
winx5 = 3
winy5 = 3

W5 = weight_variable([winx5, winy5, num_filters4, num_filters5])
b5 = bias_variable([num_filters5])
h5 = tf.nn.elu(conv2d(h4, W5, 'VALID') + b5)

# 10 X 10 X 80
# 2x2 Max pooling, no padding on edges
p2 = tf.nn.max_pool(h5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


# Input is now 5x5x80
# Need to flatten convolutional output
p2_size = np.product([s.value for s in p2.get_shape()[1:]])
p2f = tf.reshape(p2, [-1, p2_size ])

# Input is now  2000
# Dense layer
num_hidden_a = 256
W6a = tf.Variable(tf.truncated_normal([p2_size, num_hidden_a],stddev=0.1))
b6a = tf.Variable(tf.constant(random.uniform(0,1.0),shape=[num_hidden_a]))
h6a = tf.nn.elu(tf.matmul(p2f,W6a) + b6a)

# Drop out training
keep_prob = tf.placeholder("float")
h6_drop = tf.nn.dropout(h6a, keep_prob)

# Output Layer
W7 = tf.Variable(tf.truncated_normal([num_hidden_a, 10], stddev=0.1))
b7 = tf.Variable(tf.constant(0.1,shape=[10]))
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
learning_rate = tf.train.exponential_decay(.1, global_step,
                                           100.0, 0.96, staircase=True)
#learning_rate = 1e-3 # Comment this line off if you don't want fixed rate

# activate this to use adaptive gradient
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
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
    trainSize = numExamples
    batchSize = 100 # changed from 500 'cos GPU memory is limited
    numBatchesPerEpoch = trainSize//batchSize
    train_acc = np.zeros(epochs//10)
    # Compute initial loss for reference
    batch = mnist.train.next_batch(batchSize)
    loss_initial = cross_entropy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print("%6d. loss = %s" %(-1, loss_initial))
    save_per_steps = 100000//2//batchSize # Define how many steps to run before saving. Default is 50kimages
    print('Saving model every %d steps' %save_per_steps)
    f = open('trainingStatus.txt', 'wb')
    for i in xrange(numBatchesPerEpoch*epochs): #tqdm(xrange(numBatchesPerEpoch*epochs), ascii=True):
        # Set up training batch
        batch = mnist.train.next_batch(batchSize)
        # Run one iteration of training
        _, loss_val = sess.run([train_step, cross_entropy], 
                               feed_dict={x: batch[0], 
                                          y_: batch[1], 
                                          keep_prob: 0.5
                                          })        
        # Debugging lines
        if i%20==0:
            loss_initial = cross_entropy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("%6d. loss = %s" %(-1, loss_initial))
            print("%6d. loss = %s (Learning rate: %.8f)" %(i, loss_val, learning_rate.eval()))
            print(W1.eval()[0][0][0][0], W2.eval()[1][1][1][0], W3.eval()[2][2][2][0])
        # Record accuracy & save checkpoint
        if (i % save_per_steps== 0) & (i>0) :            
            # Check accuracy on train set
            train_acc = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("Training acc: %.5f" %train_acc)
            # Save variables checkpoint
            print("Saving model checkpoint..")
            saver.save(sess, "conv2a_partial_Mnist.ckpt")
        # Save checkpoint AND remove previous checkpoint to save space (~150MB per file). 
        # Done on every epoch
        if i%(100000//batchSize)==0 & i>0:
            saver.save(sess, "conv_"+str(i//(100000//batchSize))+"Mnist.ckpt")
            if (i>=(100000//batchSize)):
                try:
                    os.remove("conv_"+str(i//(100000//batchSize) - 1)+"Mnist.ckpt")
                except:
                    pass
    # Save the weights after all the training has been done
    saver.save(sess, "Mnist.ckpt")
                   
