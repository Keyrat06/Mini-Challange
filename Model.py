import numpy as np
import random
import tensorflow as tf
import math
from scipy import ndimage

regularizable_para = 0

regularizable_para = 0

#-------------------HELPER FUNCTIONS-------------#
<<<<<<< HEAD
def _two_element_tuple(int_or_tuple):
    """Converts `int_or_tuple` to height, width.
    
    Several of the functions that follow accept arguments as either
    a tuple of 2 integers or a single integer.  A single integer
    indicates that the 2 values of the tuple are the same.
    
    This functions normalizes the input value by always returning a tuple.
    
    Args:
      int_or_tuple: A list of 2 ints, a single int or a tf.TensorShape.
    
    Returns:
      A tuple with 2 values.
    
    Raises:
      ValueError: If `int_or_tuple` it not well formed.
    """
    if isinstance(int_or_tuple, (list, tuple)):
        if len(int_or_tuple) != 2:
            raise ValueError('Must be a list with 2 elements: %s' % int_or_tuple)
        return int(int_or_tuple[0]), int(int_or_tuple[1])
    if isinstance(int_or_tuple, int):
        return int(int_or_tuple), int(int_or_tuple)
    if isinstance(int_or_tuple, tf.TensorShape):
        if len(int_or_tuple) == 2:
            return int_or_tuple[0], int_or_tuple[1]
    raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                   'length 2')
    
def flatten(inputs, name=None):
    dims = inputs.get_shape()[1:]
    k = dims.num_elements()
    return tf.reshape(inputs, [-1, k])
'''
    p2_size = np.product([s.value for s in inputs.get_shape()[1:]])
    return tf.reshape(inputs, [-1, p2_size ])
    '''

def max_pool(inputs, kernel_size, stride=2, padding='VALID', name=None):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    print(kernel_size)
    return tf.nn.max_pool(inputs,
                          ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding)

=======

def scrambleImages(images):
  for i in range(len(images)):
    whatToDo = random.randint(0, 3):
    if whatToDo == 0:
      pass
    elif whatToDo == 1: #flip horizontally
      images[i] = np.fliplr(images[i])
    elif whatToDo == 2: #add Noise
      row,col,ch= images[i].shape
      mean = 0
      sigma = .05
      guass = np.random.normal(mean,sigma,(row,col,ch))
      images[i] = images[i] + guass
    elif whatToDo == 3: #blur image
      images[i] = ndimage.gaussian_filter(images[i], 3)


    # Vertical flip might not be a good idea
    # elif whatToDo == 4: #flip vertically
    #   images[i] = np.flipud(images[i])


def _two_element_tuple(int_or_tuple):
  """Converts `int_or_tuple` to height, width.

  Several of the functions that follow accept arguments as either
  a tuple of 2 integers or a single integer.  A single integer
  indicates that the 2 values of the tuple are the same.

  This functions normalizes the input value by always returning a tuple.

  Args:
    int_or_tuple: A list of 2 ints, a single int or a tf.TensorShape.

  Returns:
    A tuple with 2 values.

  Raises:
    ValueError: If `int_or_tuple` it not well formed.
  """
  if isinstance(int_or_tuple, (list, tuple)):
    if len(int_or_tuple) != 2:
      raise ValueError('Must be a list with 2 elements: %s' % int_or_tuple)
    return int(int_or_tuple[0]), int(int_or_tuple[1])
  if isinstance(int_or_tuple, int):
    return int(int_or_tuple), int(int_or_tuple)
  if isinstance(int_or_tuple, tf.TensorShape):
    if len(int_or_tuple) == 2:
      return int_or_tuple[0], int_or_tuple[1]
  raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                   'length 2')
    
def flatten(inputs, name=None):
    dims = inputs.get_shape()[1:]
    k = dims.num_elements()
    return tf.reshape(inputs, [-1, k])
'''
    p2_size = np.product([s.value for s in inputs.get_shape()[1:]])
    return tf.reshape(inputs, [-1, p2_size ])
    '''

def max_pool(inputs, kernel_size, stride=2, padding='VALID', name=None):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    return tf.nn.max_pool(inputs,
                          ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding)

>>>>>>> master
def conv2d(inputs,
       num_filters_out,
       kernel_size,
       stride=1,
       padding='SAME',
<<<<<<< HEAD
       name = ''):
    global regularizable_para
    # Find kernel width & stride
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    # Decide num of input filters
    num_filters_in = inputs.get_shape()[-1].value
    # Create weights
    weights_shape = [kernel_h, kernel_w,
                     num_filters_in, num_filters_out]
    weights = layer_variable(weights_shape, name=(name+'weight'))
    # Create biases
    biases =layer_variable([num_filters_out])
    #biases = tf.zeros([num_filters_out], dtype=tf.float32)
    # Convolute inputs with weights
    conv = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                        padding=padding)
    # Add bias then elu
    outputs = tf.nn.elu(conv + biases)
    
    regularizable_para += tf.reduce_sum(tf.square(weights))
    return outputs

def fully_connected_layer(inputs, num_output_units, name='full_layer'):
    global regularizable_para
    # Decide num of input units
    input_size = inputs.get_shape()[1].value
    # Create connection matrix and biases
    weights = layer_variable([input_size, num_output_units], name=(name+'_weights'))
    biases = layer_variable([num_output_units])
    #biases = tf.zeros([num_output_units], dtype=tf.float32)
    # Mat-mul inputs and matrix, add bias, then elu
    outputs = tf.nn.elu(tf.nn.bias_add(tf.matmul(inputs,weights),  biases))
    
    regularizable_para += tf.reduce_sum(tf.square(weights))
    return outputs
=======
       scope=None,
       name = 'conv2d'):
    global regularizable_para
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    num_filters_in = inputs.get_shape()[-1].value
    weights_shape = [kernel_h, kernel_w,
                     num_filters_in, num_filters_out]
    size = np.prod(weights_shape[0:3])
    weights = tf.truncated_normal(weights_shape, stddev=math.sqrt(2.0/size), name=(name+'weight'))
    regularizable_para += tf.reduce_sum(tf.square(weights))
    conv = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                        padding=padding)
    biases =tf.truncated_normal([num_filters_out], stddev=math.sqrt(2.0/size))
    outputs = tf.nn.bias_add(tf.nn.elu(conv), biases)
    return outputs

def fully_connected_layer(inputs, num_output_units, keep_prob =1.0, name='full_layer'):
    global regularizable_para
    input_size = inputs.get_shape()[1].value
    weights = layer_variable([input_size, num_output_units], name=(name+'_weights'))
    regularizable_para += tf.reduce_sum(tf.square(weights))
    bias = layer_variable([num_output_units])
    output = tf.nn.elu(tf.matmul(inputs,weights) + bias)
    if keep_prob < 1.0:
        return tf.nn.dropout(output, keep_prob)
    else:
        return output
>>>>>>> master
    
def layer_variable(shape, name=None):
    size = np.product(shape[0:3])
    initial = tf.truncated_normal(shape, stddev=math.sqrt(2.0/size))
    return tf.Variable(initial, name=name)

'''
def final_avg_pool(inputs, name=None):
    layer_height = inputs.get_shape()[1].value
    layer_width = inputs.get_shape()[2].value
    y_logit = tf.nn.avg_pool(inputs, 
                             [1,layer_height,layer_width,1], 
                             [1,1,1,1], 
                             'VALID')
    tf.reshape(y_logit, [-1, y_logit.get_shape()[-1].value])
    return tf.reshape(y_logit, [-1, y_logit.get_shape()[-1].value])
'''
<<<<<<< HEAD
=======

>>>>>>> master
def avg_pool(inputs,
             kernel_size,
             stride=1,
             padding='SAME',
             scope=None,
             name = None):
    
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    
<<<<<<< HEAD
    return tf.nn.avg_pool(inputs,
=======
    return tf.nn.avg_pool(inputs, 
>>>>>>> master
                             [1,kernel_h,kernel_w,1], 
                             [1,stride_h,stride_w,1], 
                             padding = padding)

<<<<<<< HEAD
=======
# Original function written by Raoul. 
# ChiHeem: Not general enough for inception models. I will write the blocks in the main model
def generalised_naive_inception(inputs, sizeAndShapes, name=''):
    global regularizable_para
    '''
    inputs: previous layer
    sizeAndShapes: a sequence of [kernelSize, numChannel]
    '''
    num_filters_in = inputs.get_shape()[-1].value
    length = len(sizeAndShapes)
    if length%2 != 0:
        print("sorry Size and Shapes must be even since they are pairs")
        return None
    else:
        outPuts = []
        outputSize = 0
        for i in xrange(length/2):
            kernalSize = sizeAndShapes[2*i]
            kernalDepth = sizeAndShapes[2*i+1]

            totalSize = kernalSize*kernalSize*num_filters_in
            outputSize += kernalDepth

            initial = tf.truncated_normal([kernalSize,kernalSize,num_filters_in,kernalDepth], stddev=math.sqrt(2.0/totalSize))
            W = tf.Variable(initial, name=name+str(kernalSize))
            regularizable_para += tf.reduce_sum(tf.square(W))
            out = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='SAME')
            outPuts.append(out)
        return tf.concat(3, outPuts), outputSize   
>>>>>>> master

#--------------------MODEL DEFN--------------------#
def model(images, 
          keep_prob, 
          scope=''):
    
    global regularizable_para
    end_points = {}
   
    # 128 X 128 X 3
    # Conv layer 1
    end_points['conv1'] = conv2d(images, 32, [3,3], name='conv1')#
    
    # 128 X 128 X 32
<<<<<<< HEAD
    end_points['conv2'] = conv2d(end_points['conv1'], 32, [3,3], name='conv2')#
=======
    end_points['conv2'] = conv2d(end_points['conv1'], 32, [5,5], name='conv2')#
>>>>>>> master
    
    # 126 X 126 X 32
    end_points['conv3'] = conv2d(end_points['conv2'], 64, [3,3], name='conv3')#
    
    # 126 X 126 X 64
    #3x3 Max pooling, no padding on edges
    end_points['pool1'] =  max_pool(end_points['conv3'], 
                                   kernel_size = 3, 
                                   stride =2, 
                                   name='pool1')
    
    # 62 X 62 X 64
    end_points['conv4'] = conv2d(end_points['pool1'], 80, [1,1], name='conv4')#
    
    # 62 X 62 X 80 
<<<<<<< HEAD
    end_points['conv5'] = conv2d(end_points['conv4'], 128, [3,3], name='conv5')#
=======
    end_points['conv5'] = conv2d(end_points['conv4'], 192, [3,3], name='conv5')#
>>>>>>> master
    
    # 60 X 60 X 192
    # 3x3 Max pooling, no padding on edges
    end_points['pool2'] = max_pool(end_points['conv5'], 
                                   kernel_size = 3, 
                                   stride = 2, 
                                   name='pool2')
    
    net = end_points['pool2']
<<<<<<< HEAD
    
    # Auxiliary heads
    aux_logits = net
    #6x6x192
    aux_logits = avg_pool(aux_logits, [5, 5], stride=5,
                            padding='VALID')
    #6x6x128
    aux_logits = conv2d(aux_logits, 128, [1, 1])
    # Shape of feature map before the final layer.
    shape = aux_logits.get_shape()
    # 1x1x768
    aux_logits = conv2d(aux_logits, 768, [shape[1].value,shape[2].value],
                          padding='VALID')
    aux_logits = tf.reshape(aux_logits, [-1,768])
    aux_logits = tf.nn.dropout(aux_logits, 0.8)
    aux_logits = flatten(aux_logits)
    aux_logits = fully_connected_layer(aux_logits,
                                       100,
                                       name = 'auxiliary_layer')
    end_points['aux_logits'] = aux_logits
    
    # 30 x 30 x 192
    # Inception block 1 #
    inception1_branch1x1 = conv2d(net, 64, [1, 1])
    inception1_branch5x5 = conv2d(net, 48, [1, 1])
    inception1_branch5x5 = conv2d(inception1_branch5x5, 64, [5, 5])
    inception1_branch3x3dbl = conv2d(net, 64, [1, 1])
    inception1_branch3x3dbl = conv2d(inception1_branch3x3dbl, 96, [3, 3])
    inception1_branch3x3dbl = conv2d(inception1_branch3x3dbl, 96, [3, 3])
    inception1_branch_pool = avg_pool(net, [3, 3])
    inception1_branch_pool = conv2d(inception1_branch_pool, 32, [1, 1])
    net = tf.concat(3, [inception1_branch1x1, inception1_branch5x5, inception1_branch3x3dbl, inception1_branch_pool])
    end_points['inception_block1'] = net
    
=======
    
    # Auxiliary heads
    aux_logits = net
    #6x6x192
    aux_logits = avg_pool(aux_logits, [5, 5], stride=5,
                            padding='VALID')
    #6x6x128
    aux_logits = conv2d(aux_logits, 128, [1, 1], scope='proj')
    # Shape of feature map before the final layer.
    shape = aux_logits.get_shape()
    # 1x1x768
    aux_logits = conv2d(aux_logits, 768, [shape[1].value,shape[2].value],
                          padding='VALID')
    aux_logits = tf.reshape(aux_logits, [-1,768])
    aux_logits = tf.nn.dropout(aux_logits, 0.8)
    aux_logits = flatten(aux_logits)
    aux_logits = fully_connected_layer(aux_logits,
                                       100,
                                       1.0,
                                       name = 'auxiliary_layer')
    end_points['aux_logits'] = aux_logits
    
    # 30 x 30 x 192
    # Inception block 1 #
    inception1_branch1x1 = conv2d(net, 64, [1, 1])
    inception1_branch5x5 = conv2d(net, 48, [1, 1])
    inception1_branch5x5 = conv2d(inception1_branch5x5, 64, [5, 5])
    inception1_branch3x3dbl = conv2d(net, 64, [1, 1])
    inception1_branch3x3dbl = conv2d(inception1_branch3x3dbl, 96, [3, 3])
    inception1_branch3x3dbl = conv2d(inception1_branch3x3dbl, 96, [3, 3])
    inception1_branch_pool = avg_pool(net, [3, 3])
    inception1_branch_pool = conv2d(inception1_branch_pool, 32, [1, 1])
    net = tf.concat(3, [inception1_branch1x1, inception1_branch5x5, inception1_branch3x3dbl, inception1_branch_pool])
    end_points['inception_block1'] = net
    
>>>>>>> master
    # 30 x 30 x 128
    # Inception block 2
    inception2_branch1x1 = conv2d(net, 64, [1, 1])
    inception2_branch5x5 = conv2d(net, 48, [1, 1])
    inception2_branch5x5 = conv2d(inception2_branch5x5, 64, [5, 5])
    inception2_branch3x3dbl = conv2d(net, 64, [1, 1])
    inception2_branch3x3dbl = conv2d(inception2_branch3x3dbl, 96, [3, 3])
    inception2_branch3x3dbl = conv2d(inception2_branch3x3dbl, 96, [3, 3])
    inception2_branch_pool = avg_pool(net, [3, 3])
    inception2_branch_pool = conv2d(inception2_branch_pool, 64, [1, 1])
    net = tf.concat(3, [inception2_branch1x1, inception2_branch5x5, inception2_branch3x3dbl, inception2_branch_pool])
    end_points['inception_block2'] = net
    
    # Input is now about 30x30x288
    # Inception block 3
    inception3_branch1x1 = conv2d(net, 96, [1, 1])
    inception3_branch7x7 = conv2d(net, 80, [1, 1])
    inception3_branch7x7 = conv2d(inception3_branch7x7, 80, [1, 7])
    inception3_branch7x7 = conv2d(inception3_branch7x7, 96, [7, 1])
    inception3_branch7x7dbl = conv2d(net, 80, [1, 1])
    inception3_branch7x7dbl = conv2d(inception3_branch7x7dbl, 80, [7, 1])
    inception3_branch7x7dbl = conv2d(inception3_branch7x7dbl, 80, [1, 7])
    inception3_branch7x7dbl = conv2d(inception3_branch7x7dbl, 80, [7, 1])
    inception3_branch7x7dbl = conv2d(inception3_branch7x7dbl, 96, [1, 7])
    inception3_branch_pool = avg_pool(net, [3, 3])
    inception3_branch_pool = conv2d(inception3_branch_pool, 96, [1, 1])
    net = tf.concat(3, [inception3_branch1x1, inception3_branch7x7, inception3_branch7x7dbl, inception3_branch_pool])
    end_points['inception_block3'] = net
    
    # Input is now about 30x30x384
    # Inception block 4
    inception4_branch3x3 = conv2d(net, 192, [3, 3], stride=2, padding='VALID')
    inception4_branch3x3dbl = conv2d(net, 32, [1, 1])
    inception4_branch3x3dbl = conv2d(inception4_branch3x3dbl, 48, [3, 3])
    inception4_branch3x3dbl = conv2d(inception4_branch3x3dbl, 48, [3, 3],
                              stride=2, padding='VALID')
    inception4_branch_pool = max_pool(net, [3, 3], stride=2, padding='VALID')
    net = tf.concat(3, [inception4_branch3x3, inception4_branch3x3dbl, inception4_branch_pool])
    end_points['inception_block4'] = net
    
    # Input is now about 15x15x768
<<<<<<< HEAD
    # Inception block 5 #
=======
    # Inception block 6 #
>>>>>>> master
    inception5_branch1x1 = conv2d(net, 192, [1, 1])
    inception5_branch7x7 = conv2d(net, 160, [1, 1])
    inception5_branch7x7 = conv2d(inception5_branch7x7, 160, [1, 7])
    inception5_branch7x7 = conv2d(inception5_branch7x7, 192, [7, 1])
    inception5_branch7x7dbl = conv2d(net, 160, [1, 1])
    inception5_branch7x7dbl = conv2d(inception5_branch7x7dbl, 160, [7, 1])
    inception5_branch7x7dbl = conv2d(inception5_branch7x7dbl, 160, [1, 7])
    inception5_branch7x7dbl = conv2d(inception5_branch7x7dbl, 160, [7, 1])
    inception5_branch7x7dbl = conv2d(inception5_branch7x7dbl, 192, [1, 7])
    inception5_branch_pool = avg_pool(net, [3, 3])
    inception5_branch_pool = conv2d(inception5_branch_pool, 192, [1, 1])
    net = tf.concat(3, [inception5_branch1x1, inception5_branch7x7, inception5_branch7x7dbl, inception5_branch_pool])
    end_points['inception5'] = net
    
    # Input is now about 15x15x384
<<<<<<< HEAD
    # Inception block 6
=======
    # Inception block 5
>>>>>>> master
    inception6_branch1x1 = conv2d(net, 192, [1, 1])
    inception6_branch7x7 = conv2d(net, 192, [1, 1])
    inception6_branch7x7 = conv2d(inception6_branch7x7, 192, [1, 7])
    inception6_branch7x7 = conv2d(inception6_branch7x7, 192, [7, 1])
    inception6_branch7x7dbl = conv2d(net, 96, [1, 1])
    inception6_branch7x7dbl = conv2d(inception6_branch7x7dbl, 192, [7, 1])
    inception6_branch7x7dbl = conv2d(inception6_branch7x7dbl, 192, [1, 7])
    inception6_branch7x7dbl = conv2d(inception6_branch7x7dbl, 192, [7, 1])
    inception6_branch7x7dbl = conv2d(inception6_branch7x7dbl, 192, [1, 7])
    inception6_branch_pool = avg_pool(net, [3, 3])
    inception6_branch_pool = conv2d(inception6_branch_pool, 192, [1, 1])
    net = tf.concat(3, [inception6_branch1x1, inception6_branch7x7, inception6_branch7x7dbl, inception6_branch_pool])
    end_points['inception6'] = net
    
    # Input is now about 15x15x768
<<<<<<< HEAD
    # Inception block 7
=======
    # Inception block 6
>>>>>>> master
    inception7_branch3x3 = conv2d(net, 192, [1, 1])
    inception7_branch3x3 = conv2d(inception7_branch3x3, 320, [3, 3], stride=2,
                           padding='VALID')
    inception7_branch7x7x3branch3x3 = conv2d(net, 192, [1, 1])
    inception7_branch7x7x3branch3x3 = conv2d(inception7_branch7x7x3branch3x3, 192, [1, 7])
    inception7_branch7x7x3branch3x3 = conv2d(inception7_branch7x7x3branch3x3, 192, [7, 1])
    inception7_branch7x7x3branch3x3 = conv2d(inception7_branch7x7x3branch3x3, 192, [3, 3],
                             stride=2, padding='VALID')
    inception7_inception7_branch_pool = max_pool(net, [3, 3], stride=2, padding='VALID')
    net = tf.concat(3, [inception7_branch3x3, inception7_branch7x7x3branch3x3, inception7_inception7_branch_pool])
    end_points['inception7'] = net
    
    # Input is now about 7x7x1280
<<<<<<< HEAD
    # Inception block 8
=======
    # Inception block 7
>>>>>>> master
    inception8_branch1x1 = conv2d(net, 320, [1, 1])
    inception8_branch3x3 = conv2d(net, 384, [1, 1])
    inception8_branch3x3 = tf.concat(3, [conv2d(inception8_branch3x3, 384, [1, 3]),
                            conv2d(inception8_branch3x3, 384, [3, 1])])
    inception8_branch3x3dbl = conv2d(net, 448, [1, 1])
    inception8_branch3x3dbl = conv2d(inception8_branch3x3dbl, 384, [3, 3])
    inception8_branch3x3dbl = tf.concat(3, [conv2d(inception8_branch3x3dbl, 384, [1, 3]),
                               conv2d(inception8_branch3x3dbl, 384, [3, 1])])
    inception8_branch_pool = avg_pool(net, [3, 3])
    inception8_branch_pool = conv2d(inception8_branch_pool, 192, [1, 1])
    net = tf.concat(3, [inception8_branch1x1, inception8_branch3x3, inception8_branch3x3dbl, inception8_branch_pool])
    end_points['inception8'] = net
    
    # Average pooling; force it to 1x1x1280
    shape = net.get_shape()
    net = avg_pool(net, shape[1:3], padding='VALID', scope='pool')
          
    # Input is now about 30x30x144
    # Need to flatten convolutional output
    net = tf.nn.dropout(net, keep_prob)
<<<<<<< HEAD
    net = flatten(net)
    
    # Hidden Layer
    y_logit = fully_connected_layer(net,100,name = 'output_layer')
    
    return {'y_logit':y_logit, 'end_points':end_points, 
            'regularizable_para':regularizable_para}#, 'aux_logits':aux_logits}
=======
    net = flatten(end_points['inception_block2'])
    
    # Hidden Layer
    y_logit = fully_connected_layer(net, 
                                                 100, 
                                                 1.0,
                                                 name = 'output_layer')
    
    return {'y_logit':y_logit, 'end_points':end_points, 
            'regularizable_para':regularizable_para, 'aux_logits':aux_logits}
>>>>>>> master
