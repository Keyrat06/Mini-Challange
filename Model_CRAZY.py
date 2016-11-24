import numpy as np
import tensorflow as tf
import math

#-------------------HELPER FUNCTIONS-------------#
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

'''
def conv2d(x, W, padding = 'SAME', scope=None, reuse = None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
'''
    
def flatten(inputs):
    p2_size = np.product([s.value for s in inputs.get_shape()[1:]])
    return tf.reshape(inputs, [-1, p2_size ])

def max_pool(input, win_size = 1, strides =1, padding = 'VALID'):
    return tf.nn.max_pool(input, 
                        ksize=[1, win_size, win_size, 1], 
                        strides=[1, strides, strides, 1], 
                        padding = padding)

def conv2d_new(inputs,
       num_filters_out,
       kernel_size,
       stride=1,
       padding='SAME',
       scope=None,
       name = None):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    num_filters_in = inputs.get_shape()[-1].value
    weights_shape = [kernel_h, kernel_w,
                     num_filters_in, num_filters_out]
    size = np.prod(weights_shape[0:3])
    weights = tf.truncated_normal(weights_shape, stddev=math.sqrt(2.0/size))
    conv = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                        padding=padding)
    biases =tf.truncated_normal([num_filters_out], stddev=math.sqrt(2.0/size))
    outputs = tf.nn.bias_add(tf.nn.elu(conv), biases)
    return outputs

def fully_connected_layer(input, num_output_units, keep_prob =1.0):
    input_size = input.get_shape()[1].value
    weights = layer_variable([input_size, num_output_units])
    bias = layer_variable([num_output_units])
    output = tf.nn.elu(tf.matmul(input,weights) + bias)
    if keep_prob < 1.0:
        return tf.nn.dropout(output, keep_prob)
    else:
        return output
    
def layer_variable(shape, name=None):
    size = np.product(shape[0:3])
    initial = tf.truncated_normal(shape, stddev=math.sqrt(2.0/size))
    return tf.Variable(initial, name=name)

#--------------------MODEL DEFN--------------------#
def model(images, 
          ground_truth, 
          keep_prob, 
          scope=''):
    
    end_points = {}
   
    # 128 X 128 X 3
    # Conv layer 1
    end_points['conv1'] = conv2d_new(images, 64, [3,3])
    
    # 128 X 128 X 64
    end_points['conv2'] = conv2d_new(end_points['conv1'], 64, [5,5])
    
    # 126 X 126 X 64
    end_points['conv3'] = conv2d_new(end_points['conv2'], 32, [3,3])
    
    # 126 X 126 X 32
    #3x3 Max pooling, no padding on edges
    end_points['pool1'] =  max_pool(end_points['conv3'], 
                                   win_size = 3, 
                                   strides =2)
    
    # 62 X 62 X 32
    end_points['conv4'] = conv2d_new(end_points['pool1'], 128, [3,3])
    
    # 62 X 62 X 128 
    end_points['conv5'] = conv2d_new(end_points['conv4'], 80, [3,3])
    
    # 60 X 60 X 80
    # 2x2 Max pooling, no padding on edges
    end_points['pool2'] = max_pool(end_points['conv5'], 
                                   win_size = 2, 
                                   strides =2)
    
    # Input is now about 30x30x192
    # Need to flatten convolutional output
    end_points['pool2f'] = flatten(end_points['pool2'])
    
    # Hidden Layer
    '''
    end_points['hidden_layer1'] = fully_connected_layer(end_points['pool2f'], 
                                                 1024, 
                                                 0.5)
    '''
    y_logit = fully_connected_layer(end_points['pool2f'], 
                                                 100, 
                                                 1.0)
    
    return y_logit, end_points