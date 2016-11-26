import numpy as np
import tensorflow as tf
from Model_CRAZY import model

# Define saved model
chkpt_name = 'Model.ckpt'

# Load test set. We don't need it for now (save loading time)
testData = np.load('testData.npz')
test = testData['arr_0']
names = {}
with open('labels.txt','r') as labelFile:
    for i,line in enumerate(labelFile.readlines()):
        names[i] = line.strip('\n')
        

# Use model
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
keep_prob = tf.placeholder("float")
y_logit, end_points = model(x, 1.0) # model is being used here
y_softmax = tf.nn.softmax(y_logit)

# Define predictor
model_pred5 = tf.nn.top_k(y_softmax, 5)

# Run model on images and predict
f = open('testEval.txt','wb')

with tf.Session() as sess:
    print('Evaluating...')
    # Restore trained model
    saver = tf.train.Saver()
    saver.restore(sess, chkpt_name)
    
    # Evaluate model 
    no_of_batches = len(test)//50
    for i in range(0,no_of_batches):
        batch = test[i*50:(i+1)*50]
        values, indices = sess.run(model_pred5,
                        {x: batch,
                         keep_prob: 1.0})
        for j in range(0, 50):
            f.write('test/%08d.jpg %d %d %d %d %d\n' %(j, 
                                                      indices[j][0], 
                                                      indices[j][1], 
                                                      indices[j][2], 
                                                      indices[j][3], 
                                                      indices[j][4]))

print('Done!')
