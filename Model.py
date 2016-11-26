import numpy as np
import tensorflow as tf
import math

#-------------------HELPER FUNCTIONS-------------#
def conv2d(x, W, padding = 'SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def layer_variable(shape, name=None):
    size = np.product(shape[0:3])
    initial = tf.truncated_normal(shape, stddev=math.sqrt(2.0/size))
    return tf.Variable(initial, name=name)

    
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



#--------------------MODEL DEFN--------------------#
def model(x, y_, keep_prob):
    # 128 X 128 X 3
    # Conv layer 1
    num_filters1 = 33
    winx1 = 11
    winy1 = 11
    W1 = layer_variable([winx1, winy1, 3, num_filters1], 'conv1_weight')
    b1 = layer_variable([num_filters1], 'conv1_bias')
    h1 = tf.nn.elu(conv2d(x, W1) + b1)
    
    # 128 X 128 X 33
    num_filters2 = 33
    winx2 = 7
    winy2 = 7
    W2 = layer_variable([winx2, winy2, num_filters1, num_filters2], 'conv2_weight')
    b2 = layer_variable([num_filters2], 'conv2_bias')
    h2 = tf.nn.elu(conv2d(h1, W2,'VALID') + b2)
    
    # 126 X 126 X 33
    num_filters3 = 64
    winx3 = 5
    winy3 = 5
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