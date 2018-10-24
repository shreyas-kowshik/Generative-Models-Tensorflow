import tensorflow as tf
import math

'''
Implementation of basic abstractions of tensorflow operations
'''

def get_W(shape,initializer=tf.truncated_normal_initializer(stddev=0.02),name='W'):
    W = tf.get_variable(shape=shape,name=name,initializer=initializer)
    return W

def get_bias(shape,initializer=tf.constant_initializer(0.0),name='b'):
    b = tf.get_variable(shape=shape,name=name,initializer=initializer)
    return b
    
def conv2d(x,ksize,out_ch,stride,padding='SAME',name='conv2d'):
    '''
    x : 4D input tensor in format 'NHWC'
    ksize : kernel size - a square kernel
    out_ch : output channesl
    stride : square stride value
    '''
    in_ch = x.shape[-1] # Infer the number of input channels
    strides = [1,stride,stride,1]
    
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        W = get_W(shape=[ksize,ksize,in_ch,out_ch])
        b = get_bias(shape=[out_ch])

        return tf.add(tf.nn.conv2d(x,W,strides=strides,padding=padding),b)

def D_conv2d(x,ksize,out_ch,stride,padding='SAME',name='D_conv2d'):
    '''
    Implements square filter convolutions

    ---Note---
    Here is the correct formula for computing the size of the output with tf.layers.conv2d_transpose():
    
    # Padding==Same:
    H = H1 * stride

    # Padding==Valid
    H = (H1-1) * stride + HF
    ----------
    
    where, H = output size, H1 = input size, HF = height of filter
    '''
    shape = x.get_shape().as_list()
    in_ch = shape[-1] # Infer the number of input channels
    strides = [1,stride,stride,1]
    
    batch_size_ = tf.shape(x)[0] # To get the dynamic shape of the Tensor during execution 

    if padding == 'SAME':
        output_shape = [batch_size_,shape[1]*stride,shape[2]*stride,out_ch]
    else:
        output_shape = [batch_size_,(shape[1] - 1)*stride + ksize,(shape[2] - 1)*stride + ksize,out_ch]
    
    with tf.variable_scope(name):
        W = get_W(shape=[ksize,ksize,out_ch,in_ch])
        b = get_bias(shape=[out_ch])

        return tf.add(tf.nn.conv2d_transpose(x,W,output_shape=output_shape,strides=strides,padding=padding),b)

def max_pool(x,ksize,stride,padding='SAME',name='max_pool'):
    kernel = [1,ksize,ksize,1]
    strides = [1,stride,stride,1]
    
    with tf.variable_scope(name):
        return tf.nn.max_pool(x,ksize=kernel,strides=strides,padding=padding)

def avg_pool(x,ksize,stride,padding='SAME',name='max_pool'):
    kernel = [1,ksize,ksize,1]
    strides = [1,stride,stride,1]
    
    with tf.variable_scope(name):
        return tf.nn.avg_pool(x,ksize=kernel,strides=strides,padding=padding)

def flatten(x):
    '''
    Returns a flattened Tensor of the input tensor
    '''
    return tf.contrib.layers.flatten(x)

def dense(x,out_units,name='dense'):
    '''
    A Fully Connected Layer
    
    x : Tensor of shape : [batch_size,in_units]
    '''
    in_units = x.shape[-1]
    with tf.variable_scope(name):
        W = get_W(shape=[in_units,out_units])
        b = get_bias(shape=[out_units])
        
        return tf.add(tf.matmul(x,W),b)

'''
---------------Pad-------------------------
'''
'''
Explanation of padding='VALID' and padding='SAME'
Reference : https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t

In short : VALID - adds no padding and drops the pixels where the kernel does not fit
           SAME - adds padding so that the kernel is able to cover all pixels
                  if odd number of columns are to be added, it adds them to the right
'''
def pad(x,padding,mode='CONSTANT'):
	return tf.pad(x,paddings=padding,mode=mode)

'''
---------------Normalizations--------------
'''

def bn(x,is_train,name='bn'):
    '''
    Batch Normalization
    '''
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(x,is_training=is_train,scale=True,center=True,epsilon=1e-8,updates_collections=None,decay=0.9)

'''
---------------Activations-----------------
'''
def lrelu(x,alpha=0.2,name='lrelu'):
    '''
    Leaky Relu
    '''
    with tf.variable_scope(name):
        o1 = 0.5*(1 + alpha)
        o2 = 0.5*(1 - alpha)
        return o1*x + o2*tf.abs(x)

def relu(x,name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(x)

def tanh(x,name='tanh'):
    with tf.variable_scope(name):
        return tf.nn.tanh(x)

def test():
	'''
	To run unit tests on the operations
	'''
	'''
	Tests for the above operations
	'''
	tf.reset_default_graph()
	X = tf.ones(shape=[100,28,28,1])
	X = pad(X,1)
	bn1 = bn(X,True)
	print(bn1.shape)
	c1 = conv2d(bn1,4,16,2,name='c1',padding='VALID')
	print(c1.shape)
	a1 = lrelu(c1,name='a1')
	print(a1.shape)
	p1 = max_pool(a1,2,2)
	print(p1.shape)
	
	'''
	This Deconvolution is a mirror image of it's convolution
	'''
	dc1 = D_conv2d(a1,4,1,2,name='dc1')
	print(dc1.shape)

# test()