from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import pickle

import tensorflow as tf
import numpy as np
from convnet_tf import ConvNet
import sys

import cifar10_utils
from tensorflow.examples.tutorials.mnist import input_data

from scipy import ndimage

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 500
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'


# class Logger(object):
#     def __init__(self, logfile):
#         self.terminal = sys.stdout
#         self.log = open(logfile, "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         #this flush method is needed for python 3 compatibility.
#         #this handles the flush command by doing nothing.
#         #you might want to specify some extra behavior here.
#         pass
#
# sys.stdout = Logger()


def shift(image, max_amt=0.2):
    new_img = np.copy(image)
    shape = new_img.shape
    max_x = int(shape[0] * max_amt)
    max_y = int(shape[1] * max_amt)
    x = np.random.randint(low=-max_x, high=max_x)
    y = np.random.randint(low=-max_y, high=max_y)
    return ndimage.interpolation.shift(new_img, shift=[x, y, 0])


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = ndimage.zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def zoom(image, min_zoom, max_zoom):
    modes = ['constant', 'nearest', 'reflect', 'wrap']
    mode = modes[np.random.randint(4)]
    zoom_val = np.random.uniform(min_zoom, max_zoom)
    return clipped_zoom(image, zoom_val, order=1, mode=mode)


def rotate(image, max_angle):
    modes = ['constant', 'nearest', 'reflect', 'wrap']
    mode = modes[np.random.randint(4)]
    angle = np.random.randint(-max_angle, max_angle)
    new_image = np.copy(image)
    return ndimage.rotate(new_image, angle=angle, order=1, reshape=False, mode=mode)


def flip_lr(image):
    return np.fliplr(image)


def augment_image(image):
    rot_p = 0.5
    zoom_p = 0.5
    flip_lr_p = 0.3

    if np.random.rand() < rot_p:
        image = rotate(image, 45)

    if np.random.rand() < zoom_p:
        image = zoom(image, 0.7, 1.3)

    if np.random.rand() < flip_lr_p:
        image = flip_lr(image)

    return image



def preprocess_data(batch, train_info, data_augment = False):
    batch = batch / 255
    # batch = batch / train_info.std
    if(data_augment):
        augmented_batch = np.zeros(batch.shape)
        for i in range(augmented_batch.shape[0]):
            augmented_batch[i,:] = augment_image(batch[i,:])

        return augmented_batch

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

  INFO = extract_info(dataset.train.images)

  FLAGS.learning_rate = 0.001
  FLAGS.batch_size = 128
  FLAGS.max_steps = 15000


  #Initialize Neural Network / graph
  X = tf.placeholder("float", [None] + n_input, name='X' )
  Y = tf.placeholder("float", [None, n_classes], name='labels')

  convnet = ConvNet(n_classes = n_classes)
  logits = convnet.inference(X)
  loss = convnet.loss(logits, Y)
  train_step = convnet.train_step(loss, FLAGS)
  accuracy = convnet.accuracy(logits,Y)
  is_training = convnet.is_training


  saver = tf.train.Saver()
  model_name = 'convnet_tf'
  checkpoint_path = FLAGS.checkpoint_dir + '/' +  model_name + '_' + time.strftime("%Y%m%d-%H%M")
  if not tf.gfile.Exists(checkpoint_path):
    tf.gfile.MakeDirs(checkpoint_path)



  # Initializing the variables and start the session
  init = tf.global_variables_initializer()
  summaries = tf.summary.merge_all()
  sess = tf.Session()

  sess.run(init)
  log_path = LOG_DIR_DEFAULT + '/' + model_name + '/' + model_name + '_' + time.strftime("%Y%m%d-%H%M")
  if not tf.gfile.Exists(log_path):
    tf.gfile.MakeDirs(log_path)
  writer = tf.summary.FileWriter(log_path)
  writer.add_graph(sess.graph)

  # saver.restore(sess, FLAGS.checkpoint_dir + model_path + '/' + model_name + '-10000')



  prev_accuracy = 0

  # FLAGS.max_steps = 2000
  for step in range(FLAGS.max_steps):
    x, y = dataset.train.next_batch(FLAGS.batch_size)
    x = preprocess_data(x, INFO, data_augment = True)

    #infenence, loss calcuation and train
    [logits_value, loss_value, _] = sess.run([logits, loss, train_step], feed_dict={X: x, Y: y, is_training: True })

    #Show loss/accuracy on train set (current batch)
    if step % FLAGS.print_freq == 0:
        accuracy_value = sess.run(accuracy, feed_dict = {logits: logits_value, Y: y})
        print('step %d: loss: %f, acc: %f' % (step, loss_value, accuracy_value))
        # summ = sess.run(summaries, feed_dict={accuracy: accuracy_value, loss: loss_value, X: x, is_training: True})
        # writer.add_summary(summ, global_step=step)


    #Show loss/accuracy on test set
    if step % FLAGS.eval_freq  == 0:
    # if False:
        x = preprocess_data(dataset.test.images, INFO, data_augment = False)
        y = dataset.test.labels

        # infenence, loss calcuation and accuracy
        [logits_value, loss_value, accuracy_value] = sess.run([logits, loss, accuracy],
                                                              feed_dict={X: x, Y: y, is_training: False})

        print('Test:')
        print('  step %d: loss: %f, acc: %f' % (step, loss_value, accuracy_value))

        if(accuracy_value < prev_accuracy):
            pass
            print(' Accuracy decreased, learning rate is decreased to %f', 2)
        prev_accuracy = accuracy_value

        summ = sess.run(summaries, feed_dict={accuracy: accuracy_value, loss: loss_value, X: x, is_training: False})
        writer.add_summary(summ, global_step=step)




    #Save model
    if step % FLAGS.checkpoint_freq == 0:
        save_path = saver.save(sess, checkpoint_path + '/' + model_name, global_step=step)
        print('Model saved at %s'%(save_path))
        pass

  x = preprocess_data(dataset.test.images, INFO, data_augment= False)
  y = dataset.test.labels

  # infenence, loss calcuation and accuracy
  [logits_value, loss_value, accuracy_value] = sess.run([logits, loss, accuracy],
                                                      feed_dict={X: x, Y: y, is_training: False})

  print('Test:')
  print('  final: loss: %f, acc: %f' % (loss_value, accuracy_value))
  summ = sess.run(summaries, feed_dict={accuracy: accuracy_value, loss: loss_value, X: x, is_training: False})
  writer.add_summary(summ, global_step=step)

  save_path = saver.save(sess, checkpoint_path + '/' + model_name, global_step=step)
  print('Model saved at %s' % (save_path))


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
