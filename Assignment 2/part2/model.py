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
from tensorflow.contrib import rnn

import numpy as np


def lstm_cell(n_hidden, forget_bias=1.0):
    return rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

class TextGenerationModel(object):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers, keep_prob = 0.5):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size
        self._keep_prob = keep_prob

        self._is_training = tf.placeholder(tf.bool, name = 'is_training')


        #Tensor of shape [batch_size , seq_length]
        self._inputs = tf.placeholder(tf.uint8, [None, None], name = 'inputs')

        self._targets = tf.placeholder(tf.uint8, [None, None], name = 'targets')


        self._input_one_hot = tf.one_hot(self._inputs,  self._vocab_size, name = 'inputs_one_hot')
        self._targets_one_hot = tf.one_hot(self._targets,  self._vocab_size, name = 'targets_one_hot')


        #Tensor of shape [batch_size * seq_length, vocab_size]
        self._logits_per_step = self._build_model()
        self._loss = self._compute_loss()
        self._probabilities = tf.nn.softmax(self._logits_per_step)


    def _get_logits(self, _, x):

        with tf.variable_scope('out', reuse=True):
            W = tf.get_variable("W")

            b = tf.get_variable("b")

            return tf.nn.bias_add(tf.matmul(x, W), b)


    def _build_model(self):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]

        # Define multiple lstm cells with tensorflow


        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self._lstm_num_hidden, forget_bias=1.0, state_is_tuple=False)
             for _ in range(self._lstm_num_layers)],
            state_is_tuple = False)


        # outputs, self._states = tf.nn.dynamic_rnn(stacked_lstm, self._input_one_hot,  dtype=tf.float32)


        self._initial_state = tf.placeholder(tf.float32, shape=(None, self._lstm_num_layers * 2 * self._lstm_num_hidden),
                                              name="lstm_init_value")

        # initial_state = tf.zeros([2, 2, self._batch_size, self._lstm_num_hidden],name='initial_state')
        # self._initial_state = tf.tuple(tf.zeros([1, self._batch_size, self._lstm_num_hidden]),
        #                                tf.zeros([1, self._batch_size, self._lstm_num_hidden]))

        # Get lstm cell output
        outputs, self._states = tf.nn.dynamic_rnn(stacked_lstm, self._input_one_hot, initial_state=self._initial_state,  dtype=tf.float32)

        # print(len(states))
        # print(states[0].c.get_shape().as_list())
        # print(states[1].c.get_shape().as_list())
        # print(type(states[0].h))
        # print(type(states[1].c))
        # print(type(states[1].h))
        # print(type(states[0]))
        # print(type(states[1]))

        #Tensor of shape [batch_size * seq_length, lstm_num_hidden ]
        outputs_reshaped = tf.reshape(outputs, [-1, self._lstm_num_hidden])

        #apply dropout
        outputs_reshaped = tf.cond(self.is_training,
                                   lambda: tf.nn.dropout(outputs_reshaped, self._keep_prob),
                                   lambda: outputs_reshaped)


        # print(outputs.get_shape().as_list())
        # print(outputs_reshaped.get_shape().as_list())

        #Compute the outputs -> logits fully connected layer
        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)
        with tf.variable_scope("fully_connected"):
            W = tf.get_variable("W", shape=[self._lstm_num_hidden, self._vocab_size],
                                        initializer=initializer_weights,
                                        regularizer=None)

            b = tf.get_variable("b", shape=[self._vocab_size],
                                       initializer=initializer_biases)

            network_output = tf.nn.bias_add(tf.matmul(outputs_reshaped, W) , b)

        #Tensor of shape [batch_size * seq_length, vocab_size]

        return network_output


    def _compute_loss(self):
        # Cross-entropy loss, averaged over timestep and batch

        #Tensor of shape [batch_size * seq_length, vocab_size]
        targets_reshaped = tf.reshape(self._targets_one_hot, [-1, self._vocab_size])

        with tf.name_scope('xent'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._logits_per_step, labels=targets_reshaped))

        return loss


    def generate_sentence(self, sess, sentence_length, prefix_sentence, dataset):

        prefix_ix = [dataset._char_to_ix[ch] for ch in prefix_sentence]
        prefix_ix = np.asarray([prefix_ix])
        generated_text_ix = list(prefix_ix[0])

        next_input = prefix_ix
        init_state = np.zeros(shape = (1, self._lstm_num_layers * 2 * self._lstm_num_hidden))
        feed_dict = { self.inputs: next_input, self._initial_state: init_state, self.is_training: False}

        probs, init_state = sess.run([self.probabilities, self._states], feed_dict)

        most_probable_ix = np.argmax(probs[-1])
        generated_text_ix.append((most_probable_ix))


        for _ in range(len(prefix_sentence), sentence_length):
            # 1 batch, 1 timestep (not one hot yet)
            next_input = np.asarray(most_probable_ix).reshape((1, 1))

            feed_dict = {self.inputs: next_input, self._initial_state: init_state, self.is_training: False}

            probs, init_state = sess.run([self.probabilities, self._states], feed_dict)

            most_probable_ix = np.argmax(probs[-1])

            generated_text_ix.append(most_probable_ix)


        return dataset.convert_to_string(generated_text_ix)



    def _sample(selfs, probs, no_samples, top_k):

        candidates = np.argsort(-probs)[:top_k]
        candidates_probs = probs[candidates]
        candidates_probs /= np.sum(candidates_probs)

        samples = np.random.choice(top_k, no_samples, p = candidates_probs, replace = False)

        return candidates[samples]



    def generate_sentence2(self, sess, sentence_length, prefix_sentence, dataset,
                          beam_width=1, samples_per_step=1, top_k=1):

        prefix_ix = [dataset._char_to_ix[ch] for ch in prefix_sentence]


        input =  np.asarray([prefix_ix])
        init_state = np.zeros(shape=(1, self._lstm_num_layers * 2 * self._lstm_num_hidden))
        feed_dict = {self.inputs: input, self._initial_state: init_state, self.is_training: False}

        probs, init_state = sess.run([self.probabilities, self._states], feed_dict)


        sampled_letter_ixs = self._sample(probs[-1], samples_per_step, top_k)

        open_list = []

        for letter_ix in sampled_letter_ixs:
            node = {}
            node['lstm_state'] = init_state
            node['hist'] = prefix_ix
            node['log_prob'] = np.log(probs[-1][letter_ix])
            node['letter_ix'] = letter_ix

            open_list.append(node)

        for _ in range(len(prefix_sentence), sentence_length):

            next_open_list = []

            for node in open_list:
                init_state = node['lstm_state']
                input_ix = node['letter_ix']

                # 1 batch, 1 timestep (not one hot yet)
                input = np.asarray(input_ix).reshape((1, 1))

                feed_dict = {self.inputs: input, self._initial_state: init_state, self.is_training: False}

                probs, final_state = sess.run([self.probabilities, self._states], feed_dict)


                #Sample next letters and add them to the list of states for the next layer
                sampled_letter_ixs = self._sample(probs[-1], samples_per_step, top_k)

                for letter_ix in sampled_letter_ixs:
                    new_node = {}
                    new_node['lstm_state'] = final_state
                    new_node['hist'] = node['hist'] + [input_ix]
                    new_node['log_prob'] = node['log_prob'] + np.log(probs[-1][letter_ix])
                    new_node['letter_ix'] = letter_ix

                    next_open_list.append(new_node)


            #keep only the states with the 'beam_width' highest probabilities

            probs = np.asarray([node['log_prob'] for node in next_open_list])
            top_ixs = np.argsort(-probs)[:beam_width]

            open_list = [next_open_list[ix] for ix in top_ixs]


        generated_texts = []
        for node in open_list:
            hist = node['hist']

            text_ixs = hist + [node['letter_ix']]

            text = dataset.convert_to_string(text_ixs)

            generated_texts.append(text)




        return generated_texts

    @property
    def probabilities(self):
        # Returns the normalized per-step probabilities
        return self._probabilities
    @property
    def inputs(self):
        """ A 3-D float32 placeholder with shape ``. """
        return self._inputs

    @property
    def targets(self):
        """ A 2-D float32 placeholder with shape `[dynamic_duration, target_size]`. """
        return self._targets


    @property
    def loss(self):
        """ A 2-D float32 placeholder with shape `[dynamic_duration, target_size]`. """
        return self._loss


    @property
    def is_training(self):
        """ A 2-D float32 placeholder with shape ` `. """
        return self._is_training