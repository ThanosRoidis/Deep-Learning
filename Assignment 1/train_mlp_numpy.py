"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
import cifar10_utils
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DNN_HIDDEN_UNITS_DEFAULT = '100'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model on the whole test set each 100 iterations.
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)


  FLAGS.dnn_hidden_units = '100'
  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  train_cifar = False

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

  # x, y = cifar10.train.next_batch(BATCH_SIZE_DEFAULT)
  # plt.interactive(False)
  # x[0] = x[0] - np.min(x[0])
  # x[0] /= np.max(x[0])
  # plt.imshow(x[0] )
  # plt.show()

  FLAGS.weight_reg_strength = 0.001
  FLAGS.learning_rate = 0.1
  FLAGS.max_steps = 10000

  mlp = MLP(n_input, dnn_hidden_units, n_classes,
            weight_decay=FLAGS.weight_reg_strength,
            weight_scale=FLAGS.weight_init_scale)

  for step in range(FLAGS.max_steps):
    x, y = dataset.train.next_batch(FLAGS.batch_size)
    x = np.reshape(x, (-1, n_input)) / norm_const

    logits = mlp.inference(x)
    loss, full_loss = mlp.loss(logits, y)
    mlp.train_step(full_loss, FLAGS)

    if step % 100 == 0:
      x = dataset.test.images
      x = np.reshape(x, (-1, n_input)) / norm_const
      y = dataset.test.labels

      logits = mlp.inference(x)
      loss, full_loss = mlp.loss(logits, y)
      print('step %d: loss: %f, %f, acc: %f' % (step, loss, full_loss, mlp.accuracy(logits, y)))


  x = dataset.test.images / norm_const
  x = np.reshape(x, (-1, n_input))
  y = dataset.test.labels

  logits = mlp.inference(x)
  L, _ = mlp.loss(logits, y)
  print('test:', L, mlp.accuracy(logits, y))



def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

import sys

if __name__ == '__main__':
  # Command line arguments

  print(sys.version)

  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  FLAGS.dnn_hidden_units
  FLAGS.learning_rate
  FLAGS.max_steps
  FLAGS.batch_size
  FLAGS.weight_init_scale
  FLAGS.weight_reg_strength
  FLAGS.data_dir

  main()
