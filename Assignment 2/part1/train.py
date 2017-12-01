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



    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

    # Setup the model that we are going to use
    if config.model_type == 'RNN':
        print("Initializing Vanilla RNN model...")
        model = VanillaRNN(
            config.input_length - 1, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )
    else:
        print("Initializing LSTM model...")
        model = LSTM(
            config.input_length - 1, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )

    ###########################################################################
    # Implement code here.
    ###########################################################################

    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate)


    ###########################################################################
    # QUESTION: It applies gradient clipping in order to avoid the exploding gradients problem
    ###########################################################################

    grads_and_vars = optimizer.compute_gradients(model.loss)

    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables))
    ############################################################################

    ###########################################################################

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #add to the summary all the trainable variables and the loss/accuracy
    tm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in tm_variables:
        tf.summary.histogram(var.name, var)
    tf.summary.scalar('xent', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()

    summary_path = config.summary_path + "{}_{}_{}".format(config.model_type, config.input_length, config.learning_rate)
    writer = tf.summary.FileWriter(summary_path)
    writer.add_graph(sess.graph)

    ###########################################################################

    accuracy_hist = []
    for train_step in range(config.train_steps):

        palindromes = utils.generate_palindrome_batch(config.batch_size, config.input_length)
        inputs, targets = palindromes[:,:-1], palindromes[:,-1]


        # Only for time measurement of step through network
        t1 = time.time()

        [_, acc_val, loss_val] = sess.run([apply_gradients_op, model.accuracy, model.loss],feed_dict = {model.inputs: inputs, model.targets: targets})


        accuracy_hist.append(acc_val)

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

            # write summary
            res_sum = sess.run(merged_summary, feed_dict={model.loss: loss_val, model.accuracy: acc_val})

            writer.add_summary(res_sum, global_step=train_step)

            #Stop if it has achieved an average accuracy higher that 0.98 on the last 20 batches
            if len(accuracy_hist) > 20:
                avg_acc = np.mean(accuracy_hist[-20:])

                if(avg_acc > 0.98):
                    pass
                    #break

    writer.close()






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


    grid_search = False

    if not grid_search:
        train(config)

    else:
        print('Starting grid search...')
        for model_type in ['RNN', 'LSTM']:
            for input_length in [5, 10, 20, 30, 40, 50]:
                for learning_rate in [0.001, 0.025, 0.25]:
                    config.model_type = model_type
                    config.input_length = input_length
                    config.learning_rate = learning_rate

                    train(config)
