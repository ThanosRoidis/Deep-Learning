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

import tensorflow as tf


class LSTM(object):

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


        # size = [batch_size, input_length, num_classes]
        self._input_one_hot = tf.one_hot(self._inputs, num_classes)
        self._targets_one_hot = tf.one_hot(self._targets, num_classes)

        self._init_gate_vars('input_modulation', 'g', initializer_weights, initializer_biases)
        self._init_gate_vars('input_gate', 'i', initializer_weights, initializer_biases)
        self._init_gate_vars('forget_gate', 'f', initializer_weights, initializer_biases)
        self._init_gate_vars('output_gate', 'o', initializer_weights, initializer_biases)

        with tf.variable_scope('output_layer'):
            tf.get_variable("W_out", shape=[self._num_hidden, self._num_classes],
                                    initializer=initializer_weights,
                                    regularizer=None)

            tf.get_variable("b_out", shape=[self._num_classes],
                                    initializer=initializer_biases)

        self._logits = self._compute_logits()
        self._loss = self._compute_loss()
        self._accuracy = self._compute_accuracy()

    def _init_gate_vars(self, scope_name, name_prefix, initializer_weights, initializer_biases):
        with tf.variable_scope(scope_name):
            tf.get_variable("W_{}x".format(name_prefix), shape=[self._input_dim, self._num_hidden],
                                        initializer=initializer_weights,
                                        regularizer=None)

            tf.get_variable("W_{}h".format(name_prefix), shape=[self._num_hidden, self._num_hidden],
                                        initializer=initializer_weights,
                                        regularizer=None)

            tf.get_variable("b_{}".format(name_prefix), shape=[self._num_hidden],
                                       initializer=initializer_biases)






    def _get_layer(self, scope_name, name_prefix, h_prev, x):

        with tf.variable_scope(scope_name, reuse=True):
            W_x = tf.get_variable("W_{}x".format(name_prefix))

            W_h = tf.get_variable("W_{}h".format(name_prefix))

            b = tf.get_variable("b_{}".format(name_prefix))

            x_mul = tf.matmul(x, W_x)
            h_mul = tf.matmul(h_prev, W_h)

            return tf.nn.bias_add(tf.add(x_mul, h_mul), b)


    def _lstm_step(self, lstm_state_tuple, x):

        h_prev, c_prev = tf.unstack(lstm_state_tuple)

        g = tf.nn.tanh(self._get_layer('input_modulation', 'g', h_prev, x))
        i = tf.nn.sigmoid(self._get_layer('input_gate', 'i', h_prev, x))
        f = tf.nn.sigmoid(self._get_layer('forget_gate', 'f', h_prev, x))
        o = tf.nn.sigmoid(self._get_layer('output_gate', 'o', h_prev, x))

        c = g * i + c_prev * f
        # output state
        h = tf.tanh(c) * o

        return tf.stack([h,c])

    def _compute_logits(self):
        # Implement the logits for predicting the last digit in the palindrome

        #initial state for c and h
        initial_state = tf.zeros([2, self._batch_size, self._num_hidden],name='initial_state')

        rnn_inputs = tf.unstack(self._input_one_hot, axis=0)

        self._states = tf.scan(self._lstm_step, rnn_inputs, initializer=initial_state)



        with tf.variable_scope('output_layer', reuse=True):
            W_out = tf.get_variable("W_out")

            b_out = tf.get_variable("b_out")

            h = self._states[-1][0]

            logits = tf.nn.bias_add(tf.matmul(h, W_out), b_out)

        self._logits = logits

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

