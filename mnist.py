from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from semi_random import *

import gzip
import numpy as np
import math
import os
import sys
import time

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('layer_type',
                           'relu',
                           """type of units to use.""")
tf.app.flags.DEFINE_float('learning_rate',
                          0.1,
                          """initial learning rate.""")
tf.app.flags.DEFINE_float('inc_factor',
                          1.0,
                          """increase the units of the convolution
                          neural network by this factor.""")
tf.app.flags.DEFINE_string('output_filepath', '',
                           """Output filepath""")

def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(DATA_DIRECTORY):
    tf.gfile.MakeDirs(DATA_DIRECTORY)
  filepath = os.path.join(DATA_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    print('File not found, downloading %s...' % filename)
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def get_data_placeholders(train=True):
  if train:
    x_input = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    y_input = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    return x_input, y_input
  else:
    x_input = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    return x_input

conv_ch1 = int(32 * FLAGS.inc_factor)
conv_ch2 = int(64 * FLAGS.inc_factor)
fc_ch1 = int(512 * FLAGS.inc_factor)
print('conv_ch1 = %i' % conv_ch1)
print('conv_ch2 = %i' % conv_ch2)
print('fc_ch1 = %i' % fc_ch1)

def build_convnet1(x_input, layer, train=True):
  if layer == 'relu':
    convlayer_gen = lambda *x : tf.nn.relu(get_conv_layer(*x))
    fc_gen = lambda *x : tf.nn.relu(get_linear_layer(*x))
  elif layer == 'rf':
    convlayer_gen = lambda *x : tf.nn.relu(
      get_conv_layer(*x, trainable=False))
    fc_gen = lambda *x : tf.nn.relu(
      get_linear_layer(*x, trainable=False))
  else:
    exit('no such layer!')
  with tf.variable_scope('conv1'):
    relu1 = convlayer_gen(x_input, NUM_CHANNELS, conv_ch1)
    pool1 = tf.nn.max_pool(relu1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
  with tf.variable_scope('conv2'): 
    relu2 = convlayer_gen(pool1, conv_ch1, conv_ch2)
    pool2 = tf.nn.max_pool(relu2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
  pool_shape = pool2.get_shape().as_list()
  out_dim = pool_shape[1] * pool_shape[2] * pool_shape[3]
  reshape = tf.reshape(pool2, [pool_shape[0], out_dim])
  with tf.variable_scope('fc1'):
    hidden = fc_gen(reshape, out_dim, fc_ch1)
  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  if train:
    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
  with tf.variable_scope('fc2'):
    y = get_linear_layer(hidden, fc_ch1, NUM_LABELS)

  return y

def build_convnet2(x_input, layer, train=True):
  if layer == 'lsr':
    convlayer_gen = lambda *x : get_semirandom_linear_conv(*x)
    fc_gen = lambda *x : get_semirandom_linear(*x)
  elif layer == 'ssr':
    convlayer_gen = lambda *x : get_semirandom_square_conv(*x)
    fc_gen = lambda *x : get_semirandom_square(*x)
  else:
    exit('no such layer!')
  with tf.variable_scope('conv1'):
    relu1, z1 = convlayer_gen(x_input, x_input, NUM_CHANNELS, conv_ch1)
    pool1 = tf.nn.max_pool(relu1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    pool_z1 = tf.nn.max_pool(z1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
  with tf.variable_scope('conv2'): 
    relu2, z2 = convlayer_gen(pool1, pool_z1, conv_ch1, conv_ch2)
    pool2 = tf.nn.max_pool(relu2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    pool_z2 = tf.nn.max_pool(z2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
  pool_shape = pool2.get_shape().as_list()
  out_dim = pool_shape[1] * pool_shape[2] * pool_shape[3]
  reshape = tf.reshape(pool2, [pool_shape[0], out_dim])
  reshape_z = tf.reshape(pool_z2, [pool_shape[0], out_dim])
  with tf.variable_scope('fc1'):
    hidden, _ = fc_gen(reshape, reshape_z, out_dim, fc_ch1)
  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  if train:
    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
  with tf.variable_scope('fc2'):
    y = get_linear_layer(hidden, fc_ch1, NUM_LABELS)

  return y

def build_convnet(x_input, layer, train=True):
  if layer == 'lsr' or layer == 'ssr':
    return build_convnet2(x_input, layer, train)
  else:
    return build_convnet1(x_input, layer, train)

def main(_):
  layer_type = FLAGS.layer_type
  eta = FLAGS.learning_rate

  output_filepath = FLAGS.output_filepath

  print('Now training mnist with %s units and eta = %.4f' % (layer_type, eta))
  print('output_filepath = %s' % output_filepath)
  print()

  # Get the data.
  train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
  train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
  test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
  test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

  # Extract it into np arrays.
  train_data = extract_data(train_data_filename, 60000)
  train_labels = extract_labels(train_labels_filename, 60000)
  test_data = extract_data(test_data_filename, 10000)
  test_labels = extract_labels(test_labels_filename, 10000)

  # Generate a validation set.
  validation_data = train_data[:VALIDATION_SIZE, ...]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_data = train_data[VALIDATION_SIZE:, ...]
  train_labels = train_labels[VALIDATION_SIZE:]
  num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  train_data_node, train_labels_node = get_data_placeholders(True)
  eval_input = get_data_placeholders(False)

  with tf.variable_scope('model') as scope:
    logits = build_convnet(train_data_node, layer_type, True)
    scope.reuse_variables()
    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(build_convnet(eval_input, layer_type, False))

  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node, logits=logits))

  batch = tf.Variable(0, dtype=tf.float32)
  learning_rate = tf.train.exponential_decay(
      eta,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_input: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_input: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a np array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the optimizer to update weights.
      sess.run(optimizer, feed_dict=feed_dict)
      # print some extra information once reach the evaluation frequency
      if step % EVAL_FREQUENCY == 0:
        # fetch some extra nodes' data
        l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                      feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
        if np.isnan(l):
          break
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)

    with open(output_filepath, 'w') as f:
      f.write('%f\n' % test_error)


if __name__ == '__main__':
  tf.app.run()
