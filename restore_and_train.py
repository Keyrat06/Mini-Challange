import random
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from Model_BigInception_BN import model
from scrambleImages import scrambleImages
#import matplotlib.pyplot as plt

# Set parameters
np.random.seed(0)
tf.set_random_seed(0)
batchSize = 70
epochs = 30 # Epoch here is defined to be 100k images
toSave = False
chkpt_name = 'conv2a_partial.ckpt'

# Load train data
trainData = np.load('trainData.npz')
train = trainData['arr_0']
trainlabels = trainData['arr_1']
train = train.astype('float32')

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

# -------------------- SETUP UP ACTUAL TRAINING ---------------
# Use model
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y_ = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder("float")
packer = model(x, keep_prob) # model is being used here
y_logit = packer['y_logit']
end_points = packer['end_points']
regularizable_para = packer['regularizable_para']

# Define accuracy for evaluation purposes
y_softmax = tf.nn.softmax(y_logit)
_ , model_pred1 = tf.nn.top_k(y_softmax, 1)
_ , model_pred5 = tf.nn.top_k(y_softmax, 5)
correct1 = tf.reduce_any(tf.equal(model_pred1, tf.expand_dims(y_, 1)), 1)
correct5 = tf.reduce_any(tf.equal(model_pred5, tf.expand_dims(y_, 1)), 1)
accuracy1 = tf.reduce_sum(tf.cast(correct1, tf.float32))/batchSize
accuracy5 = tf.reduce_sum(tf.cast(correct5, tf.float32))/batchSize

# Set loss function (cross entropy)
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    y_logit, #logits 
                                    y_       #actual class labels
                                ))#+0.01*regularizable_para

#Set learning rate
global_step = tf.Variable(0.0, trainable=False)
''' Activate either one for exponential decay/constant rate '''
learning_rate = tf.train.exponential_decay(1e-5, global_step,
                                           200.0, 0.96, staircase=True)
#learning_rate = 2.5e-5 # Comment this line off if you don't want fixed rate

''' Activate this to use adaptive gradient '''
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
     
# Restore session
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, chkpt_name)
    sess.run(tf.initialize_variables([global_step])) # reset global_step
    
    trainSize = len(train)
    validSize = len(valid)
    numBatchesPerEpoch = trainSize//batchSize

    # Initialize
    best_loss = sess.run(cross_entropy,
                    feed_dict={ x: train[0:100000:2000],
                                y_: trainlabels[0:100000:2000],
                                keep_prob: 1.0
                                });
    loss_val = best_loss
    last_i = 0
    train_acc5 = 0
    
    # Compute number of steps before saving
    save_per_steps = 100000//2//batchSize # Define how many steps to run before saving. Default is 50k images
    print('Saving model every %d steps' %save_per_steps)
    
    # Open file to write training status
    f = open('trainingStatus.txt', 'a')
    f2 = open('trainingLoss.txt', 'a')
          
    for i in range(19265, numBatchesPerEpoch*epochs):
        # Set up training batch
        batch = random.sample(range(trainSize),batchSize)
        trainBatch = train[batch]
        trainLabelBatch = trainlabels[batch]
        
        # Run one iteration of training
        if train_acc5<0.1:
            _, loss_val = sess.run([train_step, cross_entropy],
                                   feed_dict={x: trainBatch, 
                                              y_: np.transpose(trainLabelBatch), 
                                              keep_prob: 0.5
                                              })
        else:
            _, loss_val = sess.run([train_step, cross_entropy],
                                   feed_dict={x: scrambleImages(trainBatch), 
                                              y_: np.transpose(trainLabelBatch), 
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

        if i%50==0:
            # Train data
            train_acc1, train_acc5 = \
            sess.run([accuracy1, accuracy5],
                     {x: trainBatch,
                      y_: trainLabelBatch, 
                      keep_prob: 1.0})
            
            
            # These print the predicted labels and actual label as a np_array
            '''
            train_acc1, train_acc5, train_pred5 = \
            sess.run([accuracy1, accuracy5, model_pred5],
                     {x: trainBatch,
                      y_: np.transpose(trainLabelBatch), 
                      keep_prob: 1.0})
            temp = np.concatenate((train_pred5[0:50:10], np.transpose([trainLabelBatch[0:50:10]])), axis=1)
            print(temp) 
            '''
            
            # Valid data
            validBatch = random.sample(range(validSize),batchSize)
            validAcc1, validAcc5 = \
            sess.run([accuracy1, accuracy5],
                     {x: valid[validBatch],
                      y_: validlabels[validBatch],
                      keep_prob: 1.0})
            try:
                print("%6d. loss = %s (lr: %g) acc: %.5f / %.5f | %.5f / %.5f" \
                      %(i, loss_val, learning_rate.eval(), train_acc1, train_acc5, validAcc1, validAcc5))
            except:
                print("%6d. loss = %s (lr: %g) acc: %.5f / %.5f | %.5f / %.5f" \
                      %(i, loss_val, learning_rate,train_acc1,train_acc5, validAcc1, validAcc5))
            # Write to file
            f2.write("%d, %.4f, %.5f, %.5f, %.5f, %.5f\n" %(i, loss_val, train_acc1, train_acc5, validAcc1, validAcc5))
            
        # Record accuracy & save checkpoint
        if (i % save_per_steps == 0) & (i>0):
            # Save variables checkpoint
            print("Saving model checkpoint..")
            try:
                saver.save(sess, "conv2a_partial.ckpt")
                print("Model saved!")
            except:
                pass
                print("Model failed to save :(")

            # Check accuracy on train batch
            train_acc1, train_acc5 = \
            sess.run([accuracy1, accuracy5],
                     {x: trainBatch,
                      y_: trainLabelBatch, 
                      keep_prob: 1.0})
            print("Training acc: %.5f /%.5f" %(train_acc1,train_acc5))
            
            # And now the validation set
            validBatchSize = 50
            batchesForValidation = len(valid)//validBatchSize
            totalAcc1 = 0
            totalAcc5 = 0
            count = 0
            for j in tqdm(range(0, batchesForValidation), ascii=True):
                validAcc1, validAcc5, validtop5 = sess.run([accuracy1, accuracy5, model_pred5],
                                            {x: valid[j*validBatchSize:(j+1)*validBatchSize],
                                             y_: validlabels[j*validBatchSize:(j+1)*validBatchSize],
                                             keep_prob: 1.0})
                totalAcc1 += validAcc1*batchSize
                totalAcc5 += validAcc5*batchSize
            validation_acc1 = totalAcc1/validSize
            validation_acc5 = totalAcc5/validSize
            print("Validation acc: %.5f /%.5f" %(validation_acc1, validation_acc5))
            
            # Write to file
            f.write("%5.2f, %.6f, %.6f \n" %(i, validation_acc1, validation_acc5))
            
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
