"""
This module implements a convolutional neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer

class ConvNet(object):
  """
  This class implements a convolutional neural network in TensorFlow.
  It incorporates a certain graph model to be trained and to be used
  in inference.
  """

  def __init__(self, n_classes = 10):
    """
    Constructor for an ConvNet object. Default values should be used as hints for
    the usage of each parameter.
    Args:
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the ConvNet.
    """
    self.n_classes = n_classes

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

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=x,
      kernel_size=[5, 5],
      filters=64,
      strides = [1,1],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=[2,2])

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      kernel_size=[5, 5],
      filters=64,
      strides = [1,1],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=[2,2])

    pool2_flat = tf.contrib.layers.flatten(inputs = pool2)

    #fc1 layer
    fc1W = tf.get_variable("W1", shape=[pool2_flat.get_shape()[1].value, 384],
                    initializer=self.weight_initializer,
                    regularizer=self.weight_regularizer)
    fc1b = tf.get_variable("b1", shape=[384],
                        initializer=tf.zeros_initializer())

    fc1l = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool2_flat, fc1W), fc1b))

    #fc2 layer
    fc2W = tf.get_variable("W2", shape=[384, 192],
                           initializer=self.weight_initializer,
                           regularizer=self.weight_regularizer)
    fc2b = tf.get_variable("b2", shape=[192],
                           initializer=tf.zeros_initializer())

    fc2l = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1l, fc2W), fc2b))



    #fc3 layer
    fc3W = tf.get_variable("W3", shape=[192, self.n_classes],
                           initializer=self.weight_initializer,
                           regularizer=self.weight_regularizer)
    fc3b = tf.get_variable("b3", shape=[self.n_classes],
                           initializer=tf.zeros_initializer())

    logits = tf.nn.bias_add(tf.matmul(fc2l, fc3W), fc3b)


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

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

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


    optimizer = tf.train.AdamOptimizer(learning_rate = flags.learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = flags.learning_rate)

    train_step = optimizer.minimize(loss)

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

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

