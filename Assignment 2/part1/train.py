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
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils
from vanilla_rnn import VanillaRNN
from lstm import LSTM

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # palindromes = utils.generate_palindrome_batch(2, 5)
    # inputs, targets = palindromes[:, :-1], palindromes[:, -1]

    # _inputs = tf.placeholder(tf.uint8, shape=[2, config.input_length - 1],
    #                               name='inputs')
    # _input_one_hot = tf.one_hot(_inputs, config.num_classes)
    #
    # sess = tf.Session();
    # sess.run(tf.global_variables_initializer())
    #
    # res = sess.run(_input_one_hot, feed_dict={_inputs: inputs})
    # print(inputs)
    # print(res)
    #
    # print()
    #
    # rnn_inputs = tf.unstack(_input_one_hot, axis=0)
    # res = sess.run(rnn_inputs, feed_dict={_inputs: inputs})
    # for i in res:
    #     print(i)
    #
    #
    # exit()

    config.input_length = 30
    config.learning_rate = 0.001
    # config.model_type = 'RNN'
    config.model_type = 'LSTM'
    # config.train_steps = 10000


    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

    # Setup the model that we are going to use
    if config.model_type == 'RNN':
        print("Initializing Vanilla RNN model...")
        model = VanillaRNN(
            config.input_length, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )
    else:
        print("Initializing LSTM model...")
        model = LSTM(
            config.input_length, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )

    ###########################################################################
    # Implement code here.
    ###########################################################################

    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate)
    model.compute_logits()
    model_loss = model.compute_loss()
    model_acc = model.accuracy()


    ###########################################################################
    # Implement code here.
    ###########################################################################

    ###########################################################################
    # QUESTION: what happens here and why?
    ###########################################################################

    grads_and_vars = optimizer.compute_gradients(model_loss)

    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables))
    ############################################################################

    ###########################################################################

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())



    ###########################################################################

    for train_step in range(config.train_steps):

        palindromes = utils.generate_palindrome_batch(config.batch_size, config.input_length)

        inputs, targets = palindromes[:,:-1], palindromes[:,-1]



        # Only for time measurement of step through network
        t1 = time.time()

        [_, acc_val, loss_val] = sess.run([apply_gradients_op, model_acc, model_loss],feed_dict = {model.inputs: inputs, model.targets: targets})

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Print the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                  "Examples/Sec = {:.2f}, Accuracy = {}, Loss = {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step,
                config.train_steps, config.batch_size, examples_per_second,
                acc_val, loss_val
            ))



    palindromes = utils.generate_palindrome_batch(config.batch_size, config.input_length)
    inputs, targets = palindromes[:, :-1], palindromes[:, -1]
    preds = sess.run(tf.argmax(model.logits, 1), feed_dict={model.inputs: inputs})

    print(inputs[:10])
    print(preds[:10])


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=10, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2500, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=10.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')

    config = parser.parse_args()

    # Train the model
    train(config)




