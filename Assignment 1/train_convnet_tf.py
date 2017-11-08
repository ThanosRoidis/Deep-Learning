from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import tensorflow as tf
import numpy as np
from convnet_tf import ConvNet

import cifar10_utils
from tensorflow.examples.tutorials.mnist import input_data

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'


def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63 / 255.0)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    return image

def preprocess_data(batch, train_info, data_augment = False):
    batch = batch / 255
    # batch = batch / train_info.std
    if(data_augment):
        # batch = [_prep_data_augment(image) for image in batch]
        batch = tf.map_fn(augment_image, batch)

    return batch


def extract_info(training_data):
    """

    :param training_data: an array with the (centered) training data
    :return: information about the training data (only std so far)
    """
    #training data (already centered)
    INFO = type('INFO', (), {})()
    N = training_data.shape[0]
    INFO.std = np.sqrt((1/N) * np.sum(np.power(training_data,2), axis = 0))

    return INFO



def train():
  """
  Performs training and evaluation of ConvNet model.

  First define your graph using class ConvNet and its methods. Then define
  necessary operations such as savers and summarizers. Finally, initialize
  your model within a tf.Session and do the training.

  ---------------------------
  How to evaluate your model:
  ---------------------------
  Evaluation on test set should be conducted over full batch, i.e. 10k images,
  while it is alright to do it over minibatch for train set.

  ---------------------------------
  How often to evaluate your model:
  ---------------------------------
  - on training set every print_freq iterations
  - on test set every eval_freq iterations

  ------------------------
  Additional requirements:
  ------------------------
  Also you are supposed to take snapshots of your model state (i.e. graph,
  weights and etc.) every checkpoint_freq iterations. For this, you should
  study TensorFlow's tf.train.Saver class.
  """

  # Set the random seeds for reproducibility. DO NOT CHANGE.


  tf.set_random_seed(42)
  np.random.seed(42)

  train_cifar = True

  if train_cifar:
    dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
    n_input = 3072
    n_classes = 10
    norm_const = 1
  else:
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_input = 784
    n_classes = 10
    norm_const = 1
  INFO = extract_info(dataset.train.images)

  FLAGS.learning_rate = 0.001
  FLAGS.batch_size = 128
  FLAGS.max_steps = 15000


  #Initialize Neural Network / graph
  X = tf.placeholder("float", [None, 32,32,3] )
  Y = tf.placeholder("float", [None, n_classes])
  is_training = tf.placeholder(tf.bool)

  convnet = ConvNet(n_classes = n_classes, is_training = is_training )
  logits = convnet.inference(X)
  loss = convnet.loss(logits, Y)
  train_step = convnet.train_step(loss, FLAGS)
  accuracy = convnet.accuracy(logits,Y)

  # # Create summary variables
  # tf.summary.scalar('L', loss)
  # tf.summary.scalar('acc', accuracy)
  # writer = tf.summary.FileWriter(LOG_DIR_DEFAULT)
  # summaries = tf.summary.merge_all()

  saver = tf.train.Saver()
  model_path = FLAGS.checkpoint_dir + '/cnn_' + time.strftime("%Y%m%d-%H%M")
  model_name = 'cnn'



  # Initializing the variables and start the session
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)

  # saver.restore(sess, FLAGS.checkpoint_dir + model_path + '/' + model_name + '-10000')



  prev_accuracy = 0

  # FLAGS.max_steps = 2000
  for step in range(FLAGS.max_steps):
    x, y = dataset.train.next_batch(FLAGS.batch_size)
    x = preprocess_data(x, INFO, data_augment = False)

    #infenence, loss calcuation and train
    [logits_value, loss_value, _] = sess.run([logits, loss, train_step], feed_dict={X: x, Y: y, is_training: True })

    #Show loss/accuracy on train set (current batch)
    if step % FLAGS.print_freq == 0:
        accuracy_value = sess.run(accuracy, feed_dict = {logits: logits_value, Y: y})
        print('step %d: loss: %f, acc: %f' % (step, loss_value, accuracy_value))


    #Show loss/accuracy on test set
    if step % FLAGS.eval_freq  == 0:
    # if False:
        x = preprocess_data(dataset.test.images, INFO, data_augment = False)
        y = dataset.test.labels

        # infenence, loss calcuation and accuracy
        [logits_value, loss_value, accuracy_value] = sess.run([logits, loss, accuracy],
                                                              feed_dict={X: x, Y: y, is_training: False})

        #summ = sess.run(summaries, feed_dict={accuracy: accuracy_value, loss: loss_value})

        print('Test:')
        print('  step %d: loss: %f, acc: %f' % (step, loss_value, accuracy_value))

        if(accuracy_value < prev_accuracy):
            pass
            print(' Accuracy decreased, learning rate is decreased to %f', 2)
        prev_accuracy = accuracy_value




    #Save model
    if step % FLAGS.checkpoint_freq == 0:
        save_path = saver.save(sess, model_path + '/' + model_name, global_step=step)
        print('Model saved at %s'%(save_path))
        pass

  x = preprocess_data(dataset.test.images, INFO, data_augment= False)
  y = dataset.test.labels

  # infenence, loss calcuation and accuracy
  [logits_value, loss_value, accuracy_value] = sess.run([logits, loss, accuracy],
                                                      feed_dict={X: x, Y: y, is_training: False})

  #summ = sess.run(summaries, feed_dict={accuracy: accuracy_value, loss: loss_value})

  print('Test:')
  print('  final: loss: %f, acc: %f' % (loss_value, accuracy_value))



def initialize_folders():
  """
  Initializes all folders in FLAGS variable.
  """

  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)

  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  print_flags()

  initialize_folders()

  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
  parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
  parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
  parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
  parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
