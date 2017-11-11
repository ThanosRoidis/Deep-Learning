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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pickle
import time


from tensorflow.contrib.layers import l1_regularizer, l2_regularizer
from tensorflow.contrib.layers import xavier_initializer

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
WEIGHT_INITIALIZATION_DICT = {'xavier': xavier_initializer, # Xavier initialisation
                              'normal': tf.random_normal_initializer, # Initialization from a standard normal
                              'uniform': tf.uniform_unit_scaling_initializer, # Initialization from a uniform distribution
                             }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/contrib.layers#Regularizers
WEIGHT_REGULARIZER_DICT = {'none': None, # No regularization
                           'l1': l1_regularizer, # L1 regularization
                           'l2': l2_regularizer # L2 regularization
                          }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/nn
ACTIVATION_DICT = {'relu': tf.nn.relu, # ReLU
                   'elu': tf.nn.elu, # ELU
                   'tanh': tf.nn.tanh, #Tanh
                   'sigmoid': tf.nn.sigmoid} #Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/train
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer, # Adadelta
                  'adagrad': tf.train.AdagradOptimizer, # Adagrad
                  'adam': tf.train.AdamOptimizer, # Adam
                  'rmsprop': tf.train.RMSPropOptimizer # RMSprop
                  }

FLAGS = None


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


def initialize_parameters():
    WEIGHT_INITIALIZATION_DICT['xavier'] =  xavier_initializer()  # Xavier initialisation
    WEIGHT_INITIALIZATION_DICT['normal'] = tf.random_normal_initializer(FLAGS.weight_initialization_scale)  # Initialization from a standard normal
    WEIGHT_INITIALIZATION_DICT['uniform'] = tf.random_uniform_initializer(FLAGS.weight_initialization_scale)

    WEIGHT_REGULARIZER_DICT['none'] = None,  # No regularization
    WEIGHT_REGULARIZER_DICT['l1'] = l1_regularizer(FLAGS.weight_regularizer_strength)  # L1 regularization
    WEIGHT_REGULARIZER_DICT['l2'] = l2_regularizer(FLAGS.weight_regularizer_strength)  # L2 regularization

    ACTIVATION_DICT['relu'] = tf.nn.relu  # ReLU
    ACTIVATION_DICT['elu'] = tf.nn.elu  # ELU
    ACTIVATION_DICT['tanh'] = tf.nn.tanh  # Tanh
    ACTIVATION_DICT['sigmoid'] = tf.nn.sigmoid  # Sigmoid

    OPTIMIZER_DICT['sgd'] = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)  # Gradient Descent
    OPTIMIZER_DICT['adadelta'] = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)  # Adadelta
    OPTIMIZER_DICT['adagrad'] = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)   # Adagrad
    OPTIMIZER_DICT['adam'] = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam
    OPTIMIZER_DICT['rmsprop'] = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)  # RMSprop



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
    n_input = [32,32,3]
    n_classes = 10
    norm_const = 255
    with open(FLAGS.data_dir + '/batches.meta', 'rb') as fo:
        label_names = pickle.load(fo, encoding='bytes')
        label_names = label_names[b'label_names']
        label_names = [name.decode('UTF-8') for name in label_names]
  else:
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_input = [784]
    n_classes = 10
    norm_const = 1
    label_names = [str(i) for i in range(10)]

  #Adam: 0.001, SGD: 0.1
  FLAGS.learning_rate = 0.001
  FLAGS.max_steps = 10000

  # tf Graph input
  X = tf.placeholder("float", [None] + n_input, name='X')
  tf.summary.image('testing_images',X,4)
  Y = tf.placeholder("float", [None, n_classes], name = 'labels')
  is_training = tf.placeholder(tf.bool)

  mlp = MLP(dnn_hidden_units, n_classes,
            is_training = is_training,
            activation_fn=ACTIVATION_DICT[FLAGS.activation],
            dropout_rate = 0.5)
            # weight_initializer=WEIGHT_INITIALIZATION_DICT[FLAGS.weight_initialization],
            # weight_regularizer=WEIGHT_REGULARIZER_DICT[FLAGS.weight_regularizer])

  logits = mlp.inference(X)
  loss = mlp.loss(logits, Y)
  train_step = mlp.train_step(loss, FLAGS)
  accuracy = mlp.accuracy(logits,Y)


  # Initializing the variables
  init = tf.global_variables_initializer()

  summaries = tf.summary.merge_all()

  # Create session
  log_path = LOG_DIR_DEFAULT + '/mlp_tf/mlp_tf_' + time.strftime("%Y%m%d-%H%M")
  sess = tf.Session()
  sess.run(init)
  writer = tf.summary.FileWriter(log_path)
  writer.add_graph(sess.graph)

  for step in range(0, FLAGS.max_steps):
    x, y = dataset.train.next_batch(FLAGS.batch_size)
    x = x / norm_const

    [logits_value, loss_value, _] = sess.run([logits, loss, train_step],
                                             feed_dict={X: x, Y: y, is_training: True })

    #Print loss/accuracy on the test set
    if step % 100 == 0:
      x = dataset.test.images  / norm_const
      y = dataset.test.labels

      [logits_value, loss_value, accuracy_value] = sess.run([logits, loss, accuracy],
                                                            feed_dict={X: x, Y: y, is_training: False})

      print('step %d: loss: %f, acc: %f' % (step, loss_value, accuracy_value))

      summ = sess.run(summaries, feed_dict={accuracy: accuracy_value, loss: loss_value, X: x, is_training: False})
      writer.add_summary(summ, global_step=step)

  #Print final loss/accuracy
  x = dataset.test.images / norm_const
  y = dataset.test.labels
  [logits_value, loss_value, accuracy_value] = sess.run([logits, loss, accuracy],feed_dict={X: x, Y: y, is_training: False})
  print('Final: loss: %f, acc: %f' % (loss_value, accuracy_value))
  #Print confusion matrix
  y_pred = np.argmax(logits_value, axis = 1)
  y_true = np.argmax(y, axis = 1)
  conf_mat = confusion_matrix(y_true= y_true, y_pred= y_pred)
  # plot_confusion_matrix(conf_mat,label_names)





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




