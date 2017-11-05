"""
This module implements training and evaluation of a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import cifar10_utils
from mlp_tf import MLP
from tensorflow.examples.tutorials.mnist import input_data

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.
DNN_HIDDEN_UNITS_DEFAULT = '100'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/cifar10'


# This is the list of options for command line arguments specified below using argparse.
# Make sure that all these options are available so we can automatically test your code
# through command line arguments.

# You can check the TensorFlow API at
# https://www.tensorflow.org/programmers_guide/variables
# https://www.tensorflow.org/api_guides/python/contrib.layers#Initializers
WEIGHT_INITIALIZATION_DICT = {'xavier': None, # Xavier initialisation
                              'normal': None, # Initialization from a standard normal
                              'uniform': None, # Initialization from a uniform distribution
                             }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/contrib.layers#Regularizers
WEIGHT_REGULARIZER_DICT = {'none': None, # No regularization
                           'l1': None, # L1 regularization
                           'l2': None # L2 regularization
                          }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/nn
ACTIVATION_DICT = {'relu': None, # ReLU
                   'elu': None, # ELU
                   'tanh': None, #Tanh
                   'sigmoid': None} #Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/train
OPTIMIZER_DICT = {'sgd': None, # Gradient Descent
                  'adadelta': None, # Adadelta
                  'adagrad': None, # Adagrad
                  'adam': None, # Adam
                  'rmsprop': None # RMSprop
                  }

FLAGS = None

def initialize_parameters():
    pass
    #OPTIMIZER_DICT['sgd'] = [1,2,3]


def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model each 100 iterations
  as you did in the task 1 of this assignment. 
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility

  tf.set_random_seed(42)
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  FLAGS.dnn_hidden_units = '300'
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  train_cifar = True

  if train_cifar:
    dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
    n_input = 3072
    n_classes = 10
    norm_const = 255
  else:
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_input = 784
    n_classes = 10
    norm_const = 1


  # tf Graph input
  X = tf.placeholder("float", [None, n_input], )
  Y = tf.placeholder("float", [None, n_classes])
  keep_prob = tf.placeholder(tf.float32)
  train_mode = tf.placeholder(tf.bool)

  mlp = MLP(dnn_hidden_units, n_classes, is_training= train_mode, dropout_rate = 0.2)

  logits = mlp.inference(X)
  loss = mlp.loss(logits, Y)
  train_step = mlp.train_step(loss, FLAGS)
  accuracy = mlp.accuracy(logits,Y)
  # print(step, L, mlp.accuracy(logits, y))

  tf.summary.scalar('L', loss)
  tf.summary.scalar('acc', accuracy)
  writer = tf.summary.FileWriter(LOG_DIR_DEFAULT)
  summaries = tf.summary.merge_all()


  # Initializing the variables
  init = tf.global_variables_initializer()

  # Create session
  sess = tf.Session()
  sess.run(init)

  FLAGS.learning_rate = 0.1
  FLAGS.max_steps = 2000
  for step in range(FLAGS.max_steps):
    x, y = dataset.train.next_batch(FLAGS.batch_size)
    x = np.reshape(x, (-1, n_input)) / norm_const

    logits_value = sess.run(logits, feed_dict={X: x, train_mode: True   })
    loss_value = sess.run(loss, feed_dict={logits: logits_value, Y: y})
    sess.run(train_step, feed_dict={X: x, Y: y, train_mode: True })
    #sess.run(train_step, feed_dict={loss: loss_value})


    if step % 100 == 0:
      x = dataset.test.images
      x = np.reshape(x, (-1, n_input)) / norm_const
      y = dataset.test.labels

      logits_value = sess.run(logits, feed_dict={X: x, train_mode: False})
      loss_value = sess.run(loss, feed_dict={logits: logits_value, Y: y})

      accuracy_value = sess.run(accuracy, feed_dict={logits: logits_value, Y: y, train_mode: False})

      summ = sess.run(summaries, feed_dict={accuracy:accuracy_value, loss: loss_value})
      writer.add_summary(summ, global_step=step)

      print('step %d: loss: %f, acc: %f' % (step, loss_value, accuracy_value))



  x = dataset.test.images / norm_const
  x = np.reshape(x, (-1, n_input))
  y = dataset.test.labels

  logits_value = sess.run(logits, feed_dict={X: x, train_mode:False})
  loss_value = sess.run(loss, feed_dict={logits: logits_value, Y: y})
  accuracy_value = sess.run(accuracy, feed_dict={logits: logits_value, Y: y, train_mode: False})

  print('Final: loss: %f, acc: %f' % (loss_value, accuracy_value))




def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  # Make directories if they do not exists yet
  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--weight_init', type = str, default = WEIGHT_INITIALIZATION_DEFAULT,
                      help='Weight initialization type [xavier, normal, uniform].')
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg', type = str, default = WEIGHT_REGULARIZER_DEFAULT,
                      help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='Dropout rate.')
  parser.add_argument('--activation', type = str, default = ACTIVATION_DEFAULT,
                      help='Activation function [relu, elu, tanh, sigmoid].')
  parser.add_argument('--optimizer', type = str, default = OPTIMIZER_DEFAULT,
                      help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
