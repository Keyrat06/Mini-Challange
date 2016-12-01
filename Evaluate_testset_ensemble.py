import numpy as np
import tensorflow as tf
from tqdm import tqdm
import Model_BigInception_BN
import Model_BigWideInception
import matplotlib.pyplot as plt
import itertools

# Define parameters
batchsize = 50

# ------------------------------------------------------------
avg_img = np.load('avg_img.npy')

######## Load test set ###################
testData = np.load('testData.npz')
test = testData['arr_0']
test = test.astype('float16')
# Subtract out average
print('Converting test set')
for i in tqdm(range(0, len(test)), ascii=True):
    test[i] = test[i]-avg_img
names = {}
with open('labels.txt','r') as labelFile:
    for i,line in enumerate(labelFile.readlines()):
        names[i] = line.strip('\n')

######## DEFINITION ######################

# Define saved models
#models = ['conv2a_partial.ckpt', 'conv2a_partial_2.ckpt']
models = [{'name': 'conv_best.ckpt', 'arch': 'BigWideInception'},
          {'name': 'conv_best_2.ckpt', 'arch': 'BigInception_BN'}]

# Use model
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y_ = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder("float")

softmax_outs = {}

packer = Model_BigWideInception.model(x, keep_prob)
softmax_outs['BigWideInception'] = tf.nn.softmax(packer['y_logit'])

packer = Model_BigInception_BN.model(x, keep_prob)
softmax_outs['BigInception_BN'] = tf.nn.softmax(packer['y_logit'])

def predict_scenes(sm):
    sm = tf.constant(sm, shape=[len(models), len(test), 100])
    mean = tf.reduce_mean(sm, 0)
    _, pred = tf.nn.top_k(mean, 5)

    with tf.Session() as sess:
        return sess.run(pred)

print('Evaluating test set')

sm = []
for m in models:    
    with tf.Session() as sess:
        # Restore trained model
        saver = tf.train.Saver([v for v in tf.all_variables() if v.name.startswith(m['arch'])])
        saver.restore(sess, m['name'])
            
        # Run model on images and predict
        no_of_batches = len(test)//batchsize
        for i in tqdm(range(0,no_of_batches), ascii=True):
            batch = test[i*batchsize:(i+1)*batchsize]
            softmax = sess.run(softmax_outs[m['arch']], {x: batch, keep_prob: 1.0})
            sm.append(list(itertools.chain.from_iterable(softmax))) 

testTop5 = predict_scenes(sm)
        
with open('testEval.txt','wb') as f, open('testEval_for_humans.txt','wb') as f2:
    for i in range(len(test)):
        f.write('test/%08d.jpg %d %d %d %d %d\n' %(i+1, 
                                                  testTop5[i][0], 
                                                  testTop5[i][1], 
                                                  testTop5[i][2], 
                                                  testTop5[i][3], 
                                                  testTop5[i][4]))
        f2.write('test/%08d.jpg %s %s %s %s %s\n' %(i+1,
                                                  names[testTop5[i][0]], 
                                                  names[testTop5[i][1]],
                                                  names[testTop5[i][2]],
                                                  names[testTop5[i][3]],
                                                  names[testTop5[i][4]],))

