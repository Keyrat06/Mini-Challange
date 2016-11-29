import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Model_BigWideInception import model
import matplotlib.pyplot as plt
import itertools

# Define parameters
batchsize = 50
testValidation = False
displayImage = False

# ------------------------------------------------------------
avg_img = np.load('avg_img.npy')
######## Load validation data ############
if (testValidation):
    validData = np.load('validData.npz')
    valid = validData['arr_0']
    validlabels = validData['arr_1']
    valid = valid.astype('float16')
    # Subtract out average 
    print('Converting validation set')
    for i in tqdm(range(0, 10000), ascii=True):
        valid[i] = valid[i]-avg_img

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
models = ['conv2a_partial.ckpt', 'conv2a_partial.ckpt']

# Use model
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y_ = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder("float")
packer = model(x, keep_prob) # model is being used here

# Unpack results
y_logit = packer['y_logit']
end_points = packer['end_points']
regularizable_para = packer['regularizable_para']

# Transform logits into softmax values
y_softmax = tf.nn.softmax(y_logit)

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
        saver = tf.train.Saver()
        saver.restore(sess, m)
            
        # Run model on images and predict
        no_of_batches = len(test)//batchsize
        for i in tqdm(range(0,no_of_batches), ascii=True):
            batch = test[i*batchsize:(i+1)*batchsize]
            softmax = sess.run(y_softmax, {x: batch, keep_prob: 1.0})
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

