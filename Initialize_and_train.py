import random
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from Model import model
#import matplotlib.pyplot as plt

# Set parameters
np.random.seed(0)
tf.set_random_seed(0)
batchSize = 70
epochs = 8 # Epoch here is defined to be 100k images


'''
def to_onehot(labels, nclasses=100):
    outlabels = np.zeros((len(labels), nclasses))
    for i, l in enumerate(labels):
        outlabels[i, l] = 1
    return outlabels
'''
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
train = trainData['arr_0'].astype('float32')
trainlabels = trainData['arr_1']

if os.path.isfile('avg_img.npy'):
    avg_img = np.load('avg_img.npy')
else:
    # Take average of all train data
    avg_img = np.zeros([128,128,3], np.float32)
    for i in tqdm(range(0, 100000), ascii=True):
        avg_img=avg_img/(i+1.0)*i + train[i]/(i+1.0)
    np.save('avg_img.npy',avg_img, False, True)
    
# Subtract out average 
for i in tqdm(range(0, 100000), ascii=True):
    train[i] = train[i]-avg_img

# Load validation data
validData = np.load('validData.npz')
valid = validData['arr_0']
validlabels = validData['arr_1']
valid = valid.astype('float16')
# Subtract out average 
for i in tqdm(range(0, 10000), ascii=True):
    valid[i] = valid[i]-avg_img

# Load test set. We don't need it for now (save loading time)
#testData = np.load('testData.npz')
#test = testData['arr_0']
# names = [""]*100
# with open('labels.txt','r') as labelFile:
#     for i,line in enumerate(labelFile.readlines()):
#         names[i] = line.strip('\n')

# -------------------- SETUP UP ACTUAL TRAINING ---------------
# Use model
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y_ = tf.placeholder(tf.int32, [None, ])
keep_prob = tf.placeholder("float")
y_logit, end_points = model(x, y_, keep_prob) # model is being used here

# Define accuracy for evaluation purposes
y_softmax = tf.nn.softmax(y_logit)
correct_prediction = tf.nn.in_top_k(y_softmax, y_, 1)
correct_prediction5 = tf.nn.in_top_k(y_softmax, y_, 5)
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
                                          keep_prob: 0.5
                                          })
                
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
            trainAcc1, trainAcc5, = sess.run([accuracy1, accuracy5], 
                                            feed_dict={x: trainBatch, 
                                                       y_: trainLabelBatch, 
                                                       keep_prob: 1.0
                                                      })
            
            validBatch = random.sample(range(validSize), batchSize)
            validAcc1, validAcc5 = sess.run([accuracy1, accuracy5], 
                                            feed_dict={x: valid[validBatch], 
                                                       y_: validlabels[validBatch], 
                                                       keep_prob: 1.0
                                                      })

            try:
                print("%6d. loss = %s (lr: %g) acc: %.5f / %.5f | %.5f / %.5f" \
                      %(i, loss_val, learning_rate.eval(), trainAcc1, trainAcc5, validAcc1, validAcc5))
            except:
                print("%6d. loss = %s (lr: %g) acc: %.5f / %.5f | %.5f / %.5f" \
                      %(i, loss_val, learning_rate, trainAcc1, trainAcc5, validAcc1, validAcc5))
                    # Write to file
            f2.write("%d, %.4f, %.5f, %.5f, %.5f, %.5f\n" %(i, loss_val, trainAcc1, trainAcc5, validAcc1, validAcc5))
            
            #print(W1.eval()[0][0][0][0], W2.eval()[1][1][1][0], W3.eval()[2][2][2][0])
            #print(tf.argmax(y_softmax.eval(feed_dict = {x:trainBatch, keep_prob: 1.0}),1).eval())
            #print(trainLabelBatch)        
            
        # Record accuracy & save checkpoint
        if (i % 490 == 0) & (i>0) :    
            # Check accuracy on train set
            trainAcc1, trainAcc5 = sess.run([accuracy1, accuracy5], 
                                            feed_dict={x: trainBatch, 
                                                       y_: trainLabelBatch, 
                                                       keep_prob: 1.0
                                                      })
            print("Training acc: %.5f /%.5f" %(trainAcc1, trainAcc5))
            
            # And now the validation set
            totalAcc1 = 0.0
            totalAcc5 = 0.0
            validBatchSize = 50
            batchesForValidation = validSize//validBatchSize
            for j in tqdm(range(0, batchesForValidation), ascii=True):
                validAcc1, validAcc5 = sess.run([accuracy1, accuracy5], 
                                                feed_dict={x: valid[j*validBatchSize:(j+1)*validBatchSize-1], 
                                                           y_: validlabels[j*validBatchSize:(j+1)*validBatchSize-1], 
                                                           keep_prob: 1.0
                                                          })
                totalAcc1 += validAcc1
                totalAcc5 += validAcc5
            validAcc1 = totalAcc1/batchesForValidation
            validAcc5 = totalAcc5/batchesForValidation
            print("Validation acc: %.5f /%.5f" %(validAcc1, validAcc5))
            
            # Write to file
            f.write("%5.2f, %.6f, %.6f \n" %(i, validAcc1, validAcc5))
            
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
