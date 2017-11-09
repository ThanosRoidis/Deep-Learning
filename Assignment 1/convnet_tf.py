"""
This module implements a convolutional neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer

import numpy as np


def augment_image(image):

  image = tf.image.random_flip_left_right(image)#, = 'rnd_flip_lr')
  image = tf.image.random_brightness(image, max_delta=63 / 255.0)#, name = 'rnd_bright')
  image = tf.image.random_contrast(image, lower=0.2, upper=1.8)#, name = 'rnd_contrast')

  # begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(image),np.asarray([0.2, 0.2, 0.5, 0.5]).reshape(1,1,4))
  # image = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox_for_draw)
  # image = tf.squeeze(image)


  return image

def batch_norm(layer_output, is_training_):
  """Applies batch normalization to the layer output.
  Args:
      layer_output: 4-d tensor, output of a FC/convolutional layer
      is_training_: placeholder or boolean variable to set to True when training
  """
  return tf.contrib.layers.batch_norm(
      layer_output,
      decay=0.999,
      center=True,
      scale=True,
      epsilon=0.001,
      activation_fn=None,
      # update moving mean and variance in place
      updates_collections=None,
      is_training=is_training_,
      reuse=None,
      # create a collections of varialbes to save
      # (moving mean and moving variance)
      variables_collections=['required_vars_collection'],
      outputs_collections=None,
      trainable=True,
      scope=None)

class ConvNet(object):
  """
  This class implements a convolutional neural network in TensorFlow.
  It incorporates a certain graph model to be trained and to be used
  in inference.
  """

  def __init__(self, n_classes):
    """
    Constructor for an ConvNet object. Default values should be used as hints for
    the usage of each parameter.
    Args:
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the ConvNet.
    """
    self.n_classes = n_classes
    self.is_training = tf.placeholder(tf.bool, name = 'is_training')
    self.dropout_rate = 0.5


  def inference(self, x):
    """
    Performs inference given an input tensor. This is the central portion
    of the network where we describe the computation graph. Here an input
    tensor undergoes a series of convolution, pooling and nonlinear operations
    as defined in this method. For the details of the model, please
    see assignment file.

    Here we recommend you to consider using variable and name scopes in order
    to make your graph more intelligible for later references in TensorBoard
    and so on. You can define a name scope for the whole model or for each
    operator group (e.g. conv+pool+relu) individually to group them by name.
    Variable scopes are essential components in TensorFlow for parameter sharing.
    Although the model(s) which are within the scope of this class do not require
    parameter sharing it is a good practice to use variable scope to encapsulate
    model.

    Args:
      x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
              the logits outputs (before softmax transformation) of the
              network. These logits can then be used with loss and accuracy
              to evaluate the model.
    """

    self.weight_initializer = xavier_initializer()
    self.weight_regularizer = l2_regularizer(0.001)

    X = tf.cond(self.is_training, lambda: tf.map_fn(augment_image, x),lambda: x)

    tf.summary.image('augmented_images', X ,10)

    # x2 = self._data_augment(x)
    # x = self._data_augment(x)

    with tf.variable_scope('conv1') as scope:
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
          inputs=X,
          kernel_size=[5, 5],
          filters=64,
          strides = [1,1],
          padding="same",
          activation=tf.nn.relu)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=[2,2])

        conv1_bn = tf.contrib.layers.batch_norm(conv1,
                                             center=True, scale=True,
                                             is_training=self.is_training)

        # conv1_bn = batch_norm(conv1, self.is_training)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1_bn, pool_size=[3, 3], strides=[2,2])

    # Convolutional Layer #2 and Pooling Layer #2

    with tf.variable_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          kernel_size=[5, 5],
          filters=64,
          strides = [1,1],
          padding="same",
          activation=tf.nn.relu)

        # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=[2,2])

        conv2_bn = tf.contrib.layers.batch_norm(conv2,
                                             center=True, scale=True,
                                             is_training=self.is_training)

        # conv2_bn = batch_norm(conv2, self.is_training)

        pool2 = tf.layers.max_pooling2d(inputs=conv2_bn, pool_size=[3, 3], strides=[2,2])

    pool2_flat = tf.contrib.layers.flatten(inputs = pool2)

    #fc1 layer
    with tf.variable_scope('fc1') as scope:
        fc1W = tf.get_variable("W1", shape=[pool2_flat.get_shape()[1].value, 384],
                        initializer=self.weight_initializer,
                        regularizer=self.weight_regularizer)
        fc1b = tf.get_variable("b1", shape=[384],
                            initializer=tf.zeros_initializer())
        # fc1l = tf.nn.bias_add(tf.matmul(pool2_flat, fc1W), fc1b)

        fc1l = tf.matmul(pool2_flat, fc1W)

        fc1l = tf.contrib.layers.batch_norm(fc1l,
                                             center=True, scale=True,
                                             is_training=self.is_training)

        # fc1l = tf.contrib.layers.batch_norm(fc1l, center=True, scale=True, is_training = self.is_training)
        fc1l = tf.nn.relu(fc1l)
        fc1l = tf.cond(self.is_training, lambda: tf.nn.dropout(fc1l, 1 - self.dropout_rate), lambda: fc1l)

    #fc2 layer
    with tf.variable_scope('fc2') as scope:
        fc2W = tf.get_variable("W2", shape=[384, 192],
                               initializer=self.weight_initializer,
                               regularizer=self.weight_regularizer)
        fc2b = tf.get_variable("b2", shape=[192],
                               initializer=tf.zeros_initializer())
        # fc2l = tf.nn.bias_add(tf.matmul(fc1l, fc2W), fc2b)

        fc2l = tf.matmul(fc1l, fc2W)

        fc2l = tf.contrib.layers.batch_norm(fc2l,
                                             center=True, scale=True,
                                             is_training=self.is_training)


        # fc2l = tf.contrib.layers.batch_norm(fc2l, center=True, scale=True, is_training = self.is_training)

        fc2l = tf.nn.relu(fc2l)
        fc2l = tf.cond(self.is_training, lambda: tf.nn.dropout(fc2l, 1 - self.dropout_rate), lambda: fc2l)

    # fc3 layer
    with tf.variable_scope('fc3') as scope:
        fc3W = tf.get_variable("W3", shape=[192, self.n_classes],
                               initializer=self.weight_initializer,
                               regularizer=self.weight_regularizer)
        fc3b = tf.get_variable("b3", shape=[self.n_classes],
                               initializer=tf.zeros_initializer())
        logits = tf.nn.bias_add(tf.matmul(fc2l, fc3W), fc3b)


        # logits = tf.contrib.layers.fully_connected(fc2l, self.n_classes,
        #                                          activation_fn=None,
        #                                          weights_initializer=self.weight_initializer,
        #                                          weights_regularizer=self.weight_regularizer,
        #                                          biases_initializer=tf.zeros_initializer(),
        #                                          biases_regularizer=None)


    return logits

  def loss(self, logits, labels):
    """
    Calculates the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    In order to implement this function you should have a look at
    tf.nn.softmax_cross_entropy_with_logits.
    
    You can use tf.summary.scalar to save scalar summaries of
    cross-entropy loss, regularization loss, and full loss (both summed)
    for use with TensorBoard. This will be useful for compiling your report.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.

    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """

    with tf.name_scope('xent'):
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

      tf.summary.scalar('L', loss)

    return loss

  def train_step(self, loss, flags):
    """
    Implements a training step using a parameters in flags.

    Args:
      loss: scalar float Tensor.
      flags: contains necessary parameters for optimization.
    Returns:
      train_step: TensorFlow operation to perform one training step
    """


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate = flags.learning_rate).minimize(loss)


    # with tf.name_scope('train'):
    #   train_step = tf.train.AdamOptimizer(learning_rate = flags.learning_rate).minimize(loss)
    #   #optimizer = tf.train.GradientDescentOptimizer(learning_rate = flags.learning_rate)


    return train_step

  def accuracy(self, logits, labels):
    """
    Calculate the prediction accuracy, i.e. the average correct predictions
    of the network.
    As in self.loss above, you can use tf.scalar_summary to save
    scalar summaries of accuracy for later use with the TensorBoard.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.

    Returns:
      accuracy: scalar float Tensor, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      tf.summary.scalar('acc', accuracy)

    return accuracy

