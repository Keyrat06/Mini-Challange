import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Model_BigInception_BN import model
import matplotlib.pyplot as plt

# Define parameters
batchsize = 50
runValidation = True
runTest = True
displayImage = False

# Define saved model
chkpt_name = 'conv2a_partial.ckpt'
#chkpt_name = 'conv_best.ckpt'

# ------------------------------------------------------------
avg_img = np.load('avg_img.npy')
######## Load validation data ############
if (runValidation):
    validData = np.load('validData.npz')
    valid = validData['arr_0']
    validlabels = validData['arr_1']
    valid = valid.astype('float16')
    # Subtract out average 
    print('Converting validation set')
    for i in tqdm(range(0, 10000), ascii=True):
        valid[i] = valid[i]-avg_img

######## Load test set ###################
if (runTest):
    testData = np.load('testData.npz')
    test = testData['arr_0']
    test = test.astype('float16')
    # Subtract out average
    print('Converting test set')
    for i in tqdm(range(0, len(test)), ascii=True):
        test[i] = test[i]-avg_img
        
######## Load labels int to string ###################    
names = {}
with open('labels.txt','r') as labelFile:
    for i,line in enumerate(labelFile.readlines()):
        names[i] = line.strip('\n')

######## DEFINITION ######################

# Use model
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y_ = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder("float")
packer = model(x, keep_prob) # model is being used here

# unpack results
y_logit = packer['y_logit']
end_points = packer['end_points']
regularizable_para = packer['regularizable_para']
# Transform logits into softmax values
y_softmax = tf.nn.softmax(y_logit)

# Define predictor
_ , model_pred1 = tf.nn.top_k(y_softmax, 1)
_ , model_pred5 = tf.nn.top_k(y_softmax, 5)
correct1 = tf.reduce_any(tf.equal(model_pred1, tf.expand_dims(y_, 1)), 1)
correct5 = tf.reduce_any(tf.equal(model_pred5, tf.expand_dims(y_, 1)), 1)
accuracy1 = tf.reduce_mean(tf.cast(correct1, tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(correct5, tf.float32))

with tf.Session() as sess:
    # Restore trained model
    saver = tf.train.Saver()
    saver.restore(sess, chkpt_name)
    
    # Test model on validation set
    if (runValidation):
        totalAcc1 = 0
        totalAcc5 = 0
        f = open('validEval.txt','wb')
        f2 = open('validEval_for_humans.txt','wb')
        print('Evaluating validation set')
        batchesForValidation = len(valid)//batchsize
        for i in tqdm(range(0, batchesForValidation), ascii=True):
            validAcc1, validAcc5, validtop5 = sess.run([accuracy1, accuracy5, model_pred5],
                                            {x: valid[i*batchsize:(i+1)*batchsize],
                                             y_: validlabels[i*batchsize:(i+1)*batchsize],
                                             keep_prob: 1.0})
            totalAcc1 += validAcc1
            totalAcc5 += validAcc5
            for j in range(0, batchsize):
                f.write('val/%08d.jpg %d %d %d %d %d\n' %(i*batchsize+j+1, 
                                                          validtop5[j][0], 
                                                          validtop5[j][1], 
                                                          validtop5[j][2], 
                                                          validtop5[j][3], 
                                                          validtop5[j][4]))
                f2.write('val/%08d.jpg %s %s %s %s %s\n' %(i*batchsize+j+1, 
                                                          names[validtop5[j][0]], 
                                                          names[validtop5[j][1]],
                                                          names[validtop5[j][2]],
                                                          names[validtop5[j][3]],
                                                          names[validtop5[j][4]],))
        print('Model accuracy over %d items: %.5f %.5f' \
              %(len(valid), totalAcc1/batchesForValidation, totalAcc5/batchesForValidation))
        f.close()
        f2.close()
        exit
        
    # Run model on images and predict
    if (runTest):
        print('Evaluating test set')
        f = open('testEval.txt','wb')
        f2 = open('testEval_for_humans.txt','wb')
        no_of_batches = len(test)//batchsize
        countItems = 0;
        for i in tqdm(range(0,no_of_batches), ascii=True):
            batch = test[i*batchsize:(i+1)*batchsize]
            testTop5 = sess.run(model_pred5,
                               {x: batch,
                                keep_prob: 1.0})
            countItems+= len(batch)
            for j in range(0, batchsize):
                if (j%batchsize==0) & displayImage:
                    plt.imshow((batch[0]+avg_img).astype('uint8'))
                    plt.show()
                    print('test/%08d.jpg %s %s %s %s %s\n' %(i*batchsize+j+1, 
                                                          names[testTop5[j][0]], 
                                                          names[testTop5[j][1]],
                                                          names[testTop5[j][2]],
                                                          names[testTop5[j][3]],
                                                          names[testTop5[j][4]],))
                    plt.pause(0.1)
                f.write('test/%08d.jpg %d %d %d %d %d\n' %(i*batchsize+j+1, 
                                                          testTop5[j][0], 
                                                          testTop5[j][1], 
                                                          testTop5[j][2], 
                                                          testTop5[j][3], 
                                                          testTop5[j][4]))
                f2.write('test/%08d.jpg %s %s %s %s %s\n' %(i*batchsize+j+1, 
                                                          names[testTop5[j][0]], 
                                                          names[testTop5[j][1]],
                                                          names[testTop5[j][2]],
                                                          names[testTop5[j][3]],
                                                          names[testTop5[j][4]],))
        print('Done! %d items processed' %countItems)
        f.close()
        f2.close()
