# MIT License
# 
# Copyright (c) 2017 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-10-19

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

################################################################################

class VanillaRNN(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)


        self._inputs = tf.placeholder(tf.uint8, shape=[self._batch_size, input_length],
                                      name='inputs')
        self._targets = tf.placeholder(tf.uint8, shape=[self._batch_size],
                                       name='targets')

        self._input_one_hot = tf.one_hot(self._inputs, num_classes)
        self._targets_one_hot = tf.one_hot(self._targets, num_classes)


        # Initialize the stuff you need
        self.W_hx = tf.get_variable("W_hx", shape=[input_dim, num_hidden],
                               initializer=initializer_weights,
                               regularizer=None)

        self.W_hh = tf.get_variable("W_hh", shape=[num_hidden, num_hidden],
                               initializer=initializer_weights,
                               regularizer=None)


        self.b_h = tf.get_variable("b_h", shape=[num_hidden],
                               initializer=initializer_biases)


        self.W_oh = tf.get_variable("W_oh", shape=[num_hidden, num_classes],
                               initializer=initializer_weights,
                               regularizer=None)

        self.b_o = tf.get_variable("b_o", shape=[num_classes],
                               initializer=initializer_biases)

        self._logits = self._compute_logits()
        self._loss = self._compute_loss()
        self._accuracy = self._compute_accuracy()



    def _rnn_step(self, h_prev, x):
        # Single step through Vanilla RNN cell ..

        x_mul = tf.matmul(x, self.W_hx)
        h_mul = tf.matmul(h_prev, self.W_hh)

        h = tf.tanh(tf.nn.bias_add(tf.add(x_mul, h_mul), self.b_h))

        return h

    def _compute_logits(self):
        # Implement the logits for predicting the last digit in the palindrome

        initial_state = tf.zeros([self._batch_size, self._num_hidden],
                                 name='initial_state')

        rnn_inputs = tf.unstack(self._input_one_hot, axis=0)

        self._states = tf.scan(self._rnn_step, rnn_inputs, initializer=initial_state)

        logits = tf.nn.bias_add(tf.matmul(self._states[-1], self.W_oh), self.b_o)

        return logits

    def _compute_loss(self):
        # Implement the cross-entropy loss for classification of the last digit

        labels = self._targets_one_hot

        with tf.name_scope('xent'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=labels))

        return loss

    def _compute_accuracy(self):
        # Implement the accuracy of predicting the
        # last digit over the current batch ...

        labels = self._targets_one_hot

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            

        return accuracy




    @property
    def inputs(self):
        """ A 3-D float32 placeholder with shape ``. """
        return self._inputs

    @property
    def targets(self):
        """ A 2-D float32 placeholder with shape `[dynamic_duration, target_size]`. """
        return self._targets


    @property
    def logits(self):
        """ A 3-D float32 placeholder with shape ``. """
        return self._logits


    @property
    def loss(self):
        """ A 3-D float32 placeholder with shape ``. """
        return self._loss


    @property
    def accuracy(self):
        """ A 3-D float32 placeholder with shape ``. """
        return self._accuracy

    @property
    def states(self):
        """ A 2-D float32 placeholder with shape ` `. """
        return self._states