import random
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import math
import matplotlib.pyplot as plt

# Set parameters
np.random.seed(0)
tf.set_random_seed(0)
batchSize = 70

#-------------------HELPER FUNCTIONS-------------#
def conv2d(x, W, padding = 'SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def layer_variable(shape, name=None):
    size = np.product(shape[0:3])
    initial = tf.truncated_normal(shape, stddev=math.sqrt(2.0/size))
    return tf.Variable(initial, name=name)

def to_onehot(labels, nclasses=100):
    outlabels = np.zeros((len(labels), nclasses))
    for i, l in enumerate(labels):
        outlabels[i, l] = 1
    return outlabels

#returns a "onelayer" convolution of input with
#each pair in sizeAndShapes
#size and shapes should be a list of [size1,number1,size2,number2,...,sizen,numbern]
#Returns (output,outputSize)
def addConvBlock(input,inputSize,sizeAndShapes,blockName=''):
    length = len(sizeAndShapes)
    if length%2 != 0:
        print "sorry Size and Shapes must be even since they are pairs"
        return None
    else:
        outPuts = []
        outputSize = 0
        for i in xrange(length/2):
            kernalSize = sizeAndShapes[2*i]
            kernalDepth = sizeAndShapes[2*i+1]

            totalSize = kernalSize*kernalSize*inputSize
            outputSize += kernalDepth

            initial = tf.truncated_normal([kernalSize,kernalSize,inputSize,kernalDepth], stddev=math.sqrt(2.0/totalSize))
            W = tf.Variable(initial, name=blackName+str(kernalSize))
            out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
            outPuts.append(out)
        return (tf.concat(3, outPuts),outputSize)


#--------------------NOTES------------------------#
'''
TO BE DONE: 
1. Store images as fp16 between 0.0 to 1.0 (saves processing time)
2. Try using elu instead of relu6 (claimed to be better)
3. Try using adaptive gradient optimizer 
'''

#--------------------LOAD DATA--------------------#

# Load train data
trainData = np.load('trainData.npz')
train = trainData['arr_0']
trainlabels = trainData['arr_1']
train = train.astype('float32')

# Take average of all train data
avg_img = np.zeros([128,128,3], np.float32)
for i in tqdm(range(0, 100000), ascii=True):
    avg_img=avg_img/(i+1.0)*i + train[i]/(i+1.0)
# Subtract out average 
for i in tqdm(range(0, 100000), ascii=True):
    train[i] = train[i]-avg_img
#train = train[0:2000]

# Load validation data
validData = np.load('validData.npz')
valid = validData['arr_0']
validlabels = validData['arr_1']
valid = valid.astype('float16')
# Subtract out average 
for i in tqdm(range(0, 10000), ascii=True):
    valid[i] = valid[i]-avg_img
#valid=valid[0:2000]

# Load test set. We don't need it for now (save loading time)
#testData = np.load('testData.npz')
#test = testData['arr_0']
# names = [""]*100
# with open('labels.txt','r') as labelFile:
#     for i,line in enumerate(labelFile.readlines()):
#         names[i] = line.strip('\n')


#--------------------MODEL DEFN--------------------#
def model(x, y_, keep_prob):
    # 128 X 128 X 3
    # Conv layer 1
    num_filters1 = 33
    winx1 = 3
    winy1 = 3
    W1 = layer_variable([winx1, winy1, 3, num_filters1], 'conv1_weight')
    b1 = layer_variable([num_filters1], 'conv1_bias')
    h1 = tf.nn.elu(conv2d(x, W1) + b1)
    
    # 128 X 128 X 33
    num_filters2 = 33
    winx2 = 3
    winy2 = 3
    W2 = layer_variable([winx2, winy2, num_filters1, num_filters2], 'conv2_weight')
    b2 = layer_variable([num_filters2], 'conv2_bias')
    h2 = tf.nn.elu(conv2d(h1, W2,'VALID') + b2)
    
    # 126 X 126 X 33
    num_filters3 = 64
    winx3 = 3
    winy3 = 3
    W3 = layer_variable([winx3, winy3, num_filters2, num_filters3], 'conv3_weight')
    b3 = layer_variable([num_filters3], 'conv3_bias')
    h3 = tf.nn.elu(conv2d(h2, W3,'SAME') + b3)
    
    # 126 X 126 X 64
    #3x3 Max pooling, no padding on edges
    p1 = tf.nn.max_pool(h3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding = 'VALID')
    
    # 62 X 62 X 64
    num_filters4 = 192
    winx4 = 1
    winy4 = 1
    W4 = layer_variable([winx4, winy4, num_filters3, num_filters4], )
    b4 = layer_variable([num_filters4])
    h4 = tf.nn.elu(conv2d(p1, W4, 'SAME') + b4)
    
    # 62 X 62 X 192 
    num_filters5 = 80
    winx5 = 3
    winy5 = 3
    W5 = layer_variable([winx5, winy5, num_filters4, num_filters5])
    b5 = layer_variable([num_filters5])
    h5 = tf.nn.elu(conv2d(h4, W5, 'VALID') + b5)
    
    # 60 X 60 X 80
    # 2x2 Max pooling, no padding on edges
    p2 = tf.nn.max_pool(h5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'VALID')
    
    # Input is now about 30x30x192
    # Need to flatten convolutional output
    p2_size = np.product([s.value for s in p2.get_shape()[1:]])
    p2f = tf.reshape(p2, [-1, p2_size ])
    
    # Dense layer 
    num_hidden_a = 1024
    W6a = layer_variable([p2_size, num_hidden_a])
    b6a = layer_variable([num_hidden_a])
    h6a = tf.nn.elu(tf.matmul(p2f,W6a) + b6a)
    
    # Drop out training
    h6_drop = tf.nn.dropout(h6a, keep_prob)
    
    # Output Layer
    W7 = layer_variable([num_hidden_a, 100])
    b7 = layer_variable([100])
    y_logit = tf.matmul(h6_drop,W7) + b7
    
    return y_logit

# -------------------- SETUP UP ACTUAL TRAINING ---------------
# Use model
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y_ = tf.placeholder(tf.int32, [None, 100])
keep_prob = tf.placeholder("float")
y_logit = model(x, y_, keep_prob) # model is being used here

# Define accuracy for evaluation purposes
y_softmax = tf.nn.softmax(y_logit)
correct_prediction = tf.nn.in_top_k(y_softmax, y_, 1)
correct_prediction5 = tf.nn.in_top_k(y_softmax, y_, 5)
#correct_prediction = tf.equal(tf.argmax(y_softmax, 1), tf.cast(y_,tf.int64))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))
# Set loss function (cross entropy)
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    y_logit, #logits 
                                    y_       #actual class labels
                                ))

#Set learning rate
global_step = tf.Variable(0.0, trainable=False)
''' Activate either one for exponential decay/constant rate '''
learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                           350.0, 0.96, staircase=True)
#learning_rate = 2.5e-5 # Comment this line off if you don't want fixed rate

''' Activate this to use adaptive gradient '''
#train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
train_step = tf.train.GradientDescentOptimizer(
     learning_rate).minimize(cross_entropy, global_step=global_step)
     
# Set up saver
saver = tf.train.Saver()

# Train!
with tf.Session() as sess:     
    # Initialize variables
    sess.run(tf.initialize_all_variables())
    
    epochs = 20 # Epoch here is defined to be 100k images
    trainSize = len(train)
    validSize = len(valid)
    numBatchesPerEpoch = trainSize//batchSize
    
    # Initialisze
    best_loss = 1e50;
    loss_val = best_loss
    last_i = 0
    
    # Compute number of steps before saving
    save_per_steps = 100000//2//batchSize # Define how many steps to run before saving. Default is 50k images
    print('Saving model every %d steps' %save_per_steps)
    
    # Open output files
    f = open('trainingStatus.txt', 'wb')
    f2 = open('trainingLoss.txt', 'wb')
          
    for i in range(numBatchesPerEpoch*epochs):
        # Set up training batch
        batch = random.sample(range(trainSize),batchSize)
        trainBatch = train[batch]
        trainLabelBatch = trainlabels[batch]
        
        # Learning schedule
        '''
        if loss_val <= 10000.0:
            learning_rate = 1e-4
        if loss_val <= 5000.0:
            learning_rate = 1e-5
        if loss_val <= 1000.0:
            learning_rate = 3e-6
        if loss_val <= 500.0:
            learning_rate = 5e-7
        if loss_val <= 100.0:
            learning_rate = 1e-8
        if loss_val <= 10.0:
            learning_rate = 1e-9
        '''
        
        # Run one iteration of training
        _, loss_val = sess.run([train_step, cross_entropy], 
                               feed_dict={x: trainBatch, 
                                          y_: trainLabelBatch, 
                                          keep_prob: 1
                                          })
        
        # Write to file
        f2.write("%d, %.4f \n" %(i, loss_val))
        
        # If we seem to have reached a good model, save it
        if (loss_val<=0.95*best_loss) & (loss_val<4.5) & (i - last_i >20):
            try:
                saver.save(sess, "conv_best.ckpt")
                print("Best model saved, loss = %.5f!" %(loss_val))
                best_loss = loss_val
                last_i=i
            except:
                print("Best model failed to save :(")
                pass

        # -- Debug --
        '''
        if math.isnan(loss_val):
            print('***')
            print(y_softmax.eval(feed_dict = {x:trainBatch, keep_prob: 1.0})) #predicted labels
            print(trainLabelBatch)  #actual labels
            print('***')
            testimg = train[batch]
            plt.imshow(testimg)
            plt.show()
            print(trainLabelBatch)
            plt.pause(5)
            print('***')
            exit
        '''
        if i%50==0:
            train_acc1 = accuracy1.eval(session=sess, feed_dict={x: trainBatch,
                y_: trainLabelBatch, keep_prob: 1.0})
            train_acc5 = accuracy5.eval(session=sess, feed_dict={x: trainBatch,
                y_: trainLabelBatch, keep_prob: 1.0})
            try:
                print("%6d. loss = %s (lr: %g) acc: %.5f / %.5f" %(i, loss_val, learning_rate.eval(),train_acc1,train_acc5))
            except:
                print("%6d. loss = %s (lr: %g) acc: %.5f / %.5f" %(i, loss_val, learning_rate,train_acc1,train_acc5))
            
            #print(W1.eval()[0][0][0][0], W2.eval()[1][1][1][0], W3.eval()[2][2][2][0])
            print(tf.argmax(y_softmax.eval(feed_dict = {x:trainBatch, keep_prob: 1.0}),1).eval())
            print(trainLabelBatch)
            #train_acc = accuracy.eval(session=sess, feed_dict={x: trainBatch,
            #    y_: trainLabelBatch, keep_prob: 1.0})
            #print("Training acc: %.5f" %train_acc)
        
            
        # Record accuracy & save checkpoint
        if (i % save_per_steps == 0) & (i>0) :    
            # Check accuracy on train set
            train_acc1 = accuracy1.eval(session=sess, feed_dict={x: trainBatch,
                y_: trainLabelBatch, keep_prob: 1.0})
            train_acc5 = accuracy5.eval(session=sess, feed_dict={x: trainBatch,
                y_: trainLabelBatch, keep_prob: 1.0})
            print("Training acc: %.5f /%.5f" %(train_acc1,train_acc5))
            
            # And now the validation set
            validBatchSize = 50
            batchesForValidation = validSize//validBatchSize
            totalAcc1 = 0
            totalAcc5 = 0
            for j in tqdm(range(0, batchesForValidation), ascii=True):
                validation_sub_acc = accuracy1.eval(session=sess,
                                                   feed_dict={x: valid[j*validBatchSize:(j+1)*validBatchSize-1],
                                                              y_: validlabels[j*validBatchSize:(j+1)*validBatchSize-1],
                                                              keep_prob: 1.0
                                                            })
                totalAcc1 += validation_sub_acc*50.0
                validation_sub_acc = accuracy5.eval(session=sess,
                                                   feed_dict={x: valid[j*validBatchSize:(j+1)*validBatchSize-1],
                                                              y_: validlabels[j*validBatchSize:(j+1)*validBatchSize-1],
                                                              keep_prob: 1.0
                                                            })
                totalAcc5 += validation_sub_acc*50.0
            validation_acc1 = totalAcc1/validSize
            validation_acc5 = totalAcc5/validSize
            print("Validation acc: %.5f /%.5f" %(validation_acc1, validation_acc5))
            
            # Write to file
            f.write("%5.2f, %.6f, %.6f \n" %(i, validation_acc1, validation_acc5))
            
            # Save variables checkpoint
            print("Saving model checkpoint..")
            try:
                saver.save(sess, "conv2a_partial.ckpt")
                print("Model saved!")
            except:
                pass
                print("Model failed to save :(")

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
