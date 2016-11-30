import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Model import model
import matplotlib.pyplot as plt



# Define saved model
chkpt_name = 'conv_best.ckpt'


# Load test set.
testData = np.load('testData.npz')
test = testData['arr_0']

# Subtract out average
avg_img = np.load('avg_img.npy')
print('Converting test set')
for i in tqdm(range(0, len(test)), ascii=True):
    test[i] = test[i]-avg_img

names = {}
with open('labels.txt','r') as labelFile:
    for i,line in enumerate(labelFile.readlines()):
        names[i] = line.strip('\n')
        

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

# # Define predictor
# _ , model_pred1 = tf.nn.top_k(y_softmax, 1)
_ , model_pred5 = tf.nn.top_k(y_softmax, 5)
# correct1 = tf.reduce_any(tf.equal(model_pred1, tf.expand_dims(y_, 1)), 1)
# correct5 = tf.reduce_any(tf.equal(model_pred5, tf.expand_dims(y_, 1)), 1)
# accuracy1 = tf.reduce_sum(tf.cast(correct1, tf.float32))/
# accuracy5 = tf.reduce_sum(tf.cast(correct5, tf.float32))/batchsize


with tf.Session() as sess:
    # Restore trained model
    saver = tf.train.Saver()
    saver.restore(sess, chkpt_name)
    # Run model on images and predict
    print('Evaluating test set')
    f = open('testEval.txt','wb')
    f2 = open('testEval_for_humans.txt','wb')

    for image in tqdm(test):
        indices = sess.run(model_pred5,
                           {x: image,
                            keep_prob: 1.0})
        f.write('test/%08d.jpg %d %d %d %d %d\n' %(i*50+j+1, 
                                                      indices[0], 
                                                      indices[1], 
                                                      indices[2], 
                                                      indices[3], 
                                                      indices[4]))
        f2.write('test/%08d.jpg %s %s %s %s %s\n' %(i*50+j+1, 
                                                      names[indices[0]], 
                                                      names[indices[1]],
                                                      names[indices[2]],
                                                      names[indices[3]],
                                                      names[indices[4]],))
    print('Done! %d items processed' %countItems)
    f.close()
    f2.close()