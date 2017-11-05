"""
This module implements a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in Tensorflow.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference, training and it
  can also be used for evaluating prediction performance.
  """

  def __init__(self, n_hidden, n_classes, is_training,
               activation_fn = tf.nn.relu, dropout_rate = 0.,
               weight_initializer = xavier_initializer(),
               weight_regularizer = l2_regularizer(0.001)):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter.

    Args:
      n_hidden: list of ints, specifies the number of units
                     in each hidden layer. If the list is empty, the MLP
                     will not have any hidden units, and the model
                     will simply perform a multinomial logistic regression.
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the MLP.
      is_training: Bool Tensor, it indicates whether the model is in training
                        mode or not. This will be relevant for methods that perform
                        differently during training and testing (such as dropout).
                        Have look at how to use conditionals in TensorFlow with
                        tf.cond.
      activation_fn: callable, takes a Tensor and returns a transformed tensor.
                          Activation function specifies which type of non-linearity
                          to use in every hidden layer.
      dropout_rate: float in range [0,1], presents the fraction of hidden units
                         that are randomly dropped for regularization.
      weight_initializer: callable, a weight initializer that generates tensors
                               of a chosen distribution.
      weight_regularizer: callable, returns a scalar regularization loss given
                               a weight variable. The returned loss will be added to
                               the total loss for training purposes.
    """
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.is_training = is_training
    self.activation_fn = activation_fn
    self.dropout_rate = dropout_rate
    self.weight_initializer = weight_initializer
    self.weight_regularizer = weight_regularizer



    return




  def inference(self, x):
    """
    Performs inference given an input tensor. This is the central portion
    of the network. Here an input tensor is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    In order to keep things uncluttered we recommend you (though it's not required)
    to implement a separate function that is used to define a fully connected
    layer of the MLP.

    In order to make your code more structured you can use variable scopes and name
    scopes. You can define a name scope for the whole model, for each hidden
    layer and for output. Variable scopes are an essential component in TensorFlow
    design for parameter sharing.

    You can use tf.summary.histogram to save summaries of the fully connected layer weights,
    biases, pre-activations, post-activations, and dropped-out activations
    for each layer. It is very useful for introspection of the network using TensorBoard.

    Args:
      x: 2D float Tensor of size [batch_size, input_dimensions]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """

    self.n_input =  x.get_shape()[1].value

    # holds [n_input, *n_hidden, n_classes]
    n_all_layers = self.n_hidden[:]
    n_all_layers.insert(0, self.n_input)
    n_all_layers.append(self.n_classes)

    keep_prob = tf.placeholder(tf.float32)

    # Store layers weight & bias
    weights = []
    biases = []
    layers = []

    for l in range(len(n_all_layers) - 1):
        print(l)

        # W = tf.Variable(tf.random_normal([n_all_layers[l], n_all_layers[l + 1]],stddev=0.0001 ))
        #b = tf.Variable(tf.random_normal([n_all_layers[l + 1]]))
        W = tf.get_variable("W" + str(l + 1), shape=[n_all_layers[l], n_all_layers[l + 1]],
                            initializer=self.weight_initializer,
                            regularizer=self.weight_regularizer)
        b = tf.get_variable("b" + str(l + 1), shape=[n_all_layers[l + 1]],
                            initializer= tf.zeros_initializer())


        if l == 0:
          layer = tf.add(tf.matmul(x, W), b)
        else:
          layer = tf.add(tf.matmul(layers[-1], W), b)


        if l != len(n_all_layers) - 2:
          layer = self.activation_fn(layer)

          #layer = tf.nn.dropout(layer, keep_prob)
          layer = tf.cond(self.is_training, lambda: tf.nn.dropout(layer, 1 - self.dropout_rate),lambda: layer)

        weights.append(W)
        biases.append(b)
        layers.append(layer)


    self.keep_prob = keep_prob
    self.weights = weights
    self.biases = biases
    self.layers = layers



    logits = layers[-1]
    return logits

  def loss(self, logits, labels):
    """
    Computes the multiclass cross-entropy loss from the logits predictions and
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
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    As in self.loss above, you can use tf.summary.scalar to save
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
