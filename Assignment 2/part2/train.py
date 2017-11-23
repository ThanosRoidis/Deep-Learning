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

import os
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

from dataset import TextDataset
from model import TextGenerationModel


def train(config):



    # a = [1,2,3]
    # b = [a[i] for i in [0,1]]
    #
    # print(a)
    # print(b)
    # exit()

    # t = np.random.rand(10, 30, 5)
    # a = tf.reshape(t, [-1, 5])
    # print(a.get_shape().as_list())
    #
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    # v = sess.run(a)

    # print(v.shape)
    # print(t[0,:,:])
    # print(v[0:30,:])

    # exit()

    # Initialize the text dataset
    config.txt_file = "books/book_EN_grimms_fairy_tails.txt"
    config.train_steps = 10000
    config.dropout_keep_prob = 1
    config.lstm_num_layers = 3

    sentence_length = 50
    prefix_sentence = 'The king'
    save_freq = 5000


    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

    dataset = TextDataset(config.txt_file)


    # Initialize the model
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers,
        keep_prob = config.dropout_keep_prob
    )


    # exit()

    ###########################################################################
    # Implement code here.
    ###########################################################################

    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate)

    # Compute the gradients for each variable
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables))

    ###########################################################################
    # Implement code here.
    ###########################################################################

    # Initialize Saver
    # saver = tf.train.Saver()
    # model_name = 'model'
    # bookname = (config.txt_file.split('/')[1]).split('.')[0]
    # checkpoint_path = './checkpoints/' + bookname + '_' + time.strftime("%Y%m%d-%H%M")
    # if not tf.gfile.Exists(checkpoint_path):
    #     tf.gfile.MakeDirs(checkpoint_path)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    generated_text = model.generate_sentence(sess, sentence_length, prefix_sentence, dataset)
    print(generated_text)

    generated_texts = model.generate_sentence2(sess, sentence_length, prefix_sentence, dataset,
                                               beam_width=4, samples_per_step=3, top_k=10)
    for text in generated_texts:
        print(text)

    # exit()

    for train_step in range(int(config.train_steps)):

        # Only for time measurement of step through network
        t1 = time.time()

        if(train_step % 2 == 0):
            x,y = dataset.batch(config.batch_size, config.seq_length)
        else:
            x, y = dataset.batch(config.batch_size, config.seq_length - 5)


        feed_dict = {}
        feed_dict[model.is_training] = True
        feed_dict[model.inputs] = x
        feed_dict[model.targets] = y
        feed_dict[model._initial_state] = np.zeros(shape = (config.batch_size, config.lstm_num_layers * 2 * config.lstm_num_hidden))

        _, loss_value = sess.run ([apply_gradients_op, model.loss], feed_dict)


        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Output the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step+1,
                int(config.train_steps), config.batch_size, examples_per_second,
                loss_value
            ))

        if(train_step % config.sample_every == 0):

            generated_text = model.generate_sentence(sess, sentence_length, prefix_sentence, dataset)
            print(generated_text)

            generated_texts = model.generate_sentence2(sess, sentence_length, prefix_sentence, dataset,
                                                       beam_width=4, samples_per_step=3, top_k=10)
            for text in generated_texts:
                print(text)



                # if(train_step % save_freq == 0):
        #     save_path = saver.save(sess, checkpoint_path + '/' + model_name, global_step=train_step)
        #     print('Model saved at %s' % (save_path))

    generated_text = model.generate_sentence(sess, sentence_length, prefix_sentence, dataset)
    print(generated_text)

    generated_texts = model.generate_sentence2(sess, sentence_length, prefix_sentence, dataset,
                                               beam_width=4, samples_per_step=3, top_k=10)
    for text in generated_texts:
        print(text)


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default = "books", required=False, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)