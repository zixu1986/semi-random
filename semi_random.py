import numpy as np
import math
import tensorflow as tf

SEED = 66478  # Set to None for random seed.

def get_conv_layer(data, ch_in, ch_out, convsize=5,
                   scale=1.0, trainable=True):
  """Get a convolution layer.

  data: input data
  ch_in: number of input channels
  ch_out: number of output channels
  convsize: size of the convolution filter
  scale: scale the stddev of the random initialization
  trainable: if the parameters are trainable
  """
  conv_input_size = convsize * convsize * ch_in
  init_stddev = scale / math.sqrt(conv_input_size)
  if trainable:
    weight_name = "weights"
    biase_name = "biases"
  else:
    weight_name = "random_weights"
    biase_name = "random_biases"
  conv_weights = tf.get_variable(
    weight_name,
    [convsize, convsize, ch_in, ch_out],
    initializer=tf.random_normal_initializer(
      stddev=init_stddev,
      seed=SEED),
    trainable=trainable)
  conv_biases = tf.get_variable(
    biase_name,
    [ch_out],
    initializer=tf.random_uniform_initializer(
      minval=-0.5*init_stddev,
      maxval=0.5*init_stddev,
      seed=SEED),
    trainable=trainable)
  conv = tf.nn.conv2d(data,
                      conv_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  return tf.nn.bias_add(conv, conv_biases)

def get_semirandom_linear_conv(h_prev, z_prev,
                               ch_in, ch_out, convsize=5):
  """Get a convolutional linear semi-random unit.
  Its form is relu(sign(r * z)) * (w * h), where
  z is the input for the random weights, h is the input for
  the adjustable weights, r is the random weight, and w is
  the adjustable weight. Note that, the sign function is implemented
  by soft sign.
  
  h_prev: input from previous layer
  z_prev: random input from previous layer
  ch_in: number of input channels
  ch_out: number of output channels
  convsize: size of the convolution filter
  """
  # Use softsign to implement sign. 
  # Use 10.0 scale to make it more like hard sign.
  z_curr = get_conv_layer(z_prev, ch_in, ch_out, convsize,
                          scale=10.0, trainable=False)
  z_curr = tf.nn.relu(tf.nn.softsign(z_curr))
  h_curr = get_conv_layer(h_prev, ch_in, ch_out, convsize)
  h_curr = z_curr * h_curr
  return h_curr, z_curr

def get_semirandom_square_conv(h_prev, z_prev,
                               ch_in, ch_out, convsize=5):
  """Get a convolutional square semi-random unit.
  Its form is relu(r * z) (w * h).

  h_prev: input from previous layer
  z_prev: random input from previous layer
  ch_in: number of input channels
  ch_out: number of output channels
  convsize: size of the convolution filter
  """
  z_curr = get_conv_layer(z_prev, ch_in, ch_out, convsize,
                          trainable=False)
  z_curr = tf.nn.relu(z_curr)
  h_curr = get_conv_layer(h_prev, ch_in, ch_out, convsize)
  h_curr = z_curr * h_curr
  return h_curr, z_curr

def get_linear_layer(x, d_in, d_out, scale=1.0, trainable=True):
  """Get a linear layer.

  x: input
  d_in: input dimension
  d_out: output dimension
  scale: scale the stddev of the random initialization
  trainable: if the parameters are trainable
  """
  init_stddev = scale / math.sqrt(d_in)
  if trainable:
    weight_name = "weights"
    biase_name = "biases"
  else:
    weight_name = "random_weights"
    biase_name = "random_biases"
  w = tf.get_variable(
    weight_name,
    [d_in, d_out],
    initializer=tf.random_normal_initializer(
      stddev=init_stddev, seed=SEED),
    trainable=trainable)
  b = tf.get_variable(
    biase_name,
    [d_out],
    initializer=tf.random_uniform_initializer(
      minval=-0.5*init_stddev,
      maxval=0.5*init_stddev,
      seed=SEED),
    trainable=trainable)
  return tf.matmul(x, w) + b

def get_semirandom_linear(h_prev, z_prev, d_in, d_out):
  """Get a linear semi-random unit.

  h_prev: input from previous layer
  z_prev: random input from the previous layer
  d_in: input dimension
  d_out: output dimension
  """
  z_curr = get_linear_layer(z_prev, d_in, d_out, 10.0, trainable=False)
  z_curr = tf.nn.relu(tf.nn.softsign(z_curr))
  h_curr = z_curr * get_linear_layer(h_prev, d_in, d_out)
  return h_curr, z_curr

def get_semirandom_square(h_prev, z_prev, d_in, d_out):
  z_curr = tf.nn.relu(get_linear_layer(z_prev, d_in, d_out, trainable=False))
  h_curr = z_curr * get_linear_layer(h_prev, d_in, d_out)
  return h_curr, z_curr
