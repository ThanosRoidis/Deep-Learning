"""
This module implements a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference, training and it
  can also be used for evaluating prediction performance.
  """

  def __init__(self, n_input, n_hidden, n_classes, weight_decay=0.0, weight_scale=0.0001):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter. Weights of the linear layers should be initialized
                      This number is required in order to specify the
                      output dimensions of the MLP.
      weight_decay: L2 regularization parameter for the weights of linear layers.
      weight_scale: scale of normal distribution to initialize weights.

    """
    self.n_input = n_input
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.weight_decay = weight_decay
    self.weight_scale = weight_scale
    
    self.W = []
    self.b = []
    
    self.f = self.ReLU
    self.f_p = self.ReLU_d
    
    n_all_layers = n_hidden[:]
    n_all_layers.insert(0,n_input)
    n_all_layers.append(n_classes)
    for layer in range(len(n_all_layers)-1):
        self.W.append(np.random.normal(size = (n_all_layers[layer], n_all_layers[layer + 1]), scale = weight_scale ))
        self.b.append(np.zeros(shape = (1,  n_all_layers[layer + 1]))) 
        
        

  def ReLU(self,x):
      return np.maximum(x,0)
  
  def ReLU_d(self,x):
    x2 = x.copy()
    x2[np.where(x2 > 0)] = 1
    x2[np.where(x2 < 0)] = 0
    return x2

  def inference(self, x):
    """
    Performs inference given an input array. This is the central portion
    of the network. Here an input tensor is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    It can be useful to save some intermediate results for easier computation of
    gradients for backpropagation during training.
    Args:
      x: 2D float array of size [batch_size, input_dimensions]

    Returns:
      logits: 2D float array of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """

   
    
    Z = [x]
    S = [x]
    for layer in range(len(self.W)):
        
        s = np.dot(Z[-1], self.W[layer]) + self.b[layer]
        S.append(s)
        
        if layer != len(self.W) - 1:
            z = self.f(s)
        else:
            z = s.copy()
        
        Z.append(z)
        
        
    self.Z = Z
    self.S = S
        
    logits = Z[-1]
    return logits


  def softmax(self, x):
      m = np.max(x, axis = 1).reshape(x.shape[0],1)
      q = np.exp(x - m)
      Z = np.sum(q, axis = 1).reshape(x.shape[0],1)
      
      return q / Z

  def softmax2(self, x):
      a = np.max(x, axis = 1).reshape(x.shape[0],1)
      logZ = a + np.log(np.sum(np.exp(x - a), axis = 1)).reshape(x.shape[0],1)
      logp = x - logZ
      p = np.exp(logp)

      return p

  def loss(self, logits, labels):
    """
    Computes the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    It can be useful to compute gradients of the loss for an easier computation of
    gradients for backpropagation during training.

    Args:
      logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int array of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.
    Returns:
      loss: scalar float, full loss = cross_entropy + reg_loss
    """

    #logq = logits #10x1
    #logZ = self.log_sum_exp(logq) # scalar
    #for all classes
    #logp = numpy.add(logq, -logZ) #10x1
    #p = numpy.exp(logp) #10x1


    n = logits.shape[0]

    p = self.softmax(logits)

    true_class = np.where(labels == 1) 
    loss = -np.sum(np.log(p[true_class])) / n
    #tmp = labels * np.log(logits) + (1-labels) * np.log(1 - logits)
    #loss = np.sum(tmp, axis = 1)


    reg_loss = 0
    for W in self.W:
        reg_loss += np.linalg.norm(W) ** 2
    reg_loss = self.weight_decay * reg_loss / (2)

    full_loss = loss + reg_loss
    
    
    d_out = p - labels;
    self.deltas = [d_out]
    

    return loss, full_loss

  def train_step(self, loss, flags):
    """
    Implements a training step using a parameters in flags.
    Use Stochastic Gradient Descent to update the parameters of the MLP.

    Args:
      loss: scalar float Tensor.
      flags: contains necessary parameters for optimization.
    Returns:

    """
    
    #Calculate deltas from the last layer to the first (no deltas for the input layer)
    for layer in range(len(self.W) - 1, 0 , -1):
        next_delta = self.deltas[0]
        
        delta =  self.f_p(self.S[layer]) * np.dot(self.W[layer], next_delta.T).T

        self.deltas.insert(0, delta)
        

    
    for layer in range(len(self.W)):
        
        s = self.Z[layer] #input to W / output of previous layer
        d = self.deltas[layer] #next layer's delta
        
        dW = np.dot(s.T , d) / d.shape[0]
        db = np.sum(d, axis = 0) / d.shape[0]

        #self.W[layer] = self.W[layer] - flags.learning_rate * dW
        #Formula for L2 reg (dW is without regularization)
        self.W[layer] = self.W[layer] - flags.learning_rate * (dW + self.weight_decay * self.W[layer])

        self.b[layer] = self.b[layer] - flags.learning_rate * db



    return

  def accuracy(self, logits, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int array of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    predicted = np.argmax(logits, axis = 1)
    true = np.argmax(labels, axis = 1)

    correct = np.sum(predicted == true)

    accuracy = correct / labels.shape[0]

    return accuracy


if __name__ == '__main__':
    # Parse arguments
     # Parse arguments
     
    
    n_input = 2
    n_classes = 2
    mlp = MLP(n_input, [3], n_classes, 0, 0.1)

         
    
    
    logits = np.asarray([[1,4,3],
                         [5,5,6]])
    
    #print(np.sum(logits, axis = 1))
    
    labels = np.asarray([[0,0,1],
                         [1,0,0]])
    
    
    
    #print(logits[np.where(labels == 1)])
    #print(np.sum(logits,axis = 1))
#==============================================================================
#     print('max', np.max(logits, axis = 1))
#     print(logits.T - np.max(logits, axis = 1))
#     print(logits - np.max(logits, axis = 1).reshape(2,1))
#==============================================================================
#==============================================================================
#     x = logits
#     m = np.max(x, axis = 1).reshape(x.shape[0],1)
#     q = np.exp(x - m)
#     Z = np.sum(q, axis = 1).reshape(x.shape[0],1)
#     
#     print(x)
#     print(m)
#     print(x - m)
#     print(q)
#     print(Z)
#     print(q / Z)
#==============================================================================


    
    batch_size = 100
    X = np.random.normal(size = (batch_size, n_input))
    labels = np.random.randint(0, n_classes, batch_size)
    labels_one_hot = np.zeros((batch_size, n_classes))
    labels_one_hot[np.arange(batch_size), labels ] = 1 
    labels = labels_one_hot

    for i in range(100):
        logits =  mlp.inference(X)
        #print(logits)
        #print(labels)
        loss, full_loss = mlp.loss(logits, labels)
        mlp.train_step(loss, [0.5])
        print(loss, mlp.accuracy(logits,labels))
    
    
    #print(np.asarray([[1,2,3],[4,5,6]]) * np.asarray([[1,2,3],[3,2,1]]))
    
    