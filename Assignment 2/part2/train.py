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
import pickle




def print_sampled_sentences(session, model, dataset):
    """generate and print sentences using hill climbing and beam search"""

    # Parameters for the generated sentences during train time
    sampled_sentence_length = 70
    # prefix_sentences = ['The king', 'Sleeping beauty is', 'THE', 'A']
    prefix_sentences = ['Thou shalt not', 'God', '88:19']
    beam_width = 8
    no_successors = 4
    top_k = 10

    all_sentences = []
    all_probs = []


    for prefix_sentence in prefix_sentences:

        # Random
        generated_texts, log_probs = model.generate_sentence(session, sampled_sentence_length, prefix_sentence,
                                                              dataset,
                                                              beam_width=1, samples_per_step=1,
                                                              top_k=10)
        print(generated_texts[0], log_probs[0])
        all_sentences.append(generated_texts[0])
        all_probs.append(log_probs[0])


        #Hill climbing
        generated_texts, log_probs = model.generate_sentence(session, sampled_sentence_length, prefix_sentence,
                                                              dataset,
                                                              beam_width=1, samples_per_step=1,
                                                              top_k=1)
        print(generated_texts[0], log_probs[0])
        all_sentences.append(generated_texts[0])
        all_probs.append(log_probs[0])


        # Beam search
        generated_texts, log_probs = model.generate_sentence(session, sampled_sentence_length, prefix_sentence, dataset,
                                                   beam_width=beam_width, samples_per_step=no_successors, top_k=top_k)
        print(generated_texts[0], log_probs[0])
        all_sentences.append(generated_texts[0])
        all_probs.append(log_probs[0])
        print()

    return all_sentences, all_probs



def load_config(config, checkpoint_path):

    # Load config
    config_dict = {}

    with open(checkpoint_path + 'config.txt', 'r') as f:
        lines = f.readlines()

        for line in lines:
            [key, value] = [item.strip() for item in line.split(':')]

            if key == 'train_steps':
                config_dict['train_steps'] = config.train_steps
                continue

            # Convert data types
            if key != 'txt_file' and key != 'summary_path':
                if key == 'log_device_placement':
                    value = (value == True)
                else:
                    if value.isdigit():
                        value = int(value)
                    else:
                        value = float(value)

            config_dict[key] = value

    return config_dict


def train(config, global_step = 0, model_folder = ""):
    """ provide a global_step and a model_folder in the ./checkpoints folder in order to
        continue training from a previously stored model
    """
    save_freq = 5000

    # If training continuous from a previous execution, read from the config file everything but the number of train steps
    if global_step != 0:
        checkpoint_path = './checkpoints/' + model_folder
        print('Continuing training from {}'.format(checkpoint_path + 'model-' + str(global_step)))

        config.__dict__ = load_config(config,checkpoint_path)

    # If a new model is trained
    else:

        # Create folder to store the model
        bookname = (config.txt_file.split('/')[1]).split('.')[0]
        model_folder = bookname + '_' + time.strftime("%Y%m%d-%H%M")
        checkpoint_path = './checkpoints/' + model_folder
        if not tf.gfile.Exists(checkpoint_path):
            tf.gfile.MakeDirs(checkpoint_path)

        # Store the configuration of the model
        file = open(checkpoint_path + '/config.txt', 'w+')
        for key, value in vars(config).items():
            file.write(key + ' : ' + str(value) + '\n')
        file.close()

        print('Training new model on {}'.format(checkpoint_path))

    # print config
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


    # Initialize Saver and session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # restore the model
    if global_step != 0:
        saver.restore(sess, checkpoint_path + 'model-' + str(global_step))

    tm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in tm_variables:
        tf.summary.histogram(var.name, var)

    tf.summary.scalar('loss', model.loss)
    sentences_placeholder = tf.placeholder(dtype=tf.string)
    tf.summary.text('Generated sentences', sentences_placeholder)
    # Store the configuration of the model


    merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(config.summary_path + model_folder)
    writer.add_graph(sess.graph)



    for train_step in range(global_step, int(config.train_steps)):

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

        # Generate and print sentences
        if(train_step % config.sample_every == 0):
            # Generate and print sentence
            all_sentences, all_probs = print_sampled_sentences(sess, model, dataset)
            writer.add_summary(
                sess.run(merged_summary, feed_dict={model.loss: loss_value, sentences_placeholder: all_sentences}),
                global_step=train_step)

            with open(config.summary_path + model_folder + '/sentences.txt', 'a+') as file:
                file.write('\n-------------------------------------------------------------\n')
                file.write(str(train_step) + '\n')
                file.write('-------------------------------------------------------------\n\n')
                for i,sentence in enumerate(all_sentences):
                    file.write(sentence + ' | ' + str(all_probs[i])  + '\n')


        # Save model
        if(train_step % save_freq == 0):
            save_path = saver.save(sess, checkpoint_path + '/' + 'model', global_step=train_step)
            print('Model saved at %s' % (save_path))

    # Generate and print sentence
    all_sentences,all_probs = print_sampled_sentences(sess, model, dataset)
    writer.add_summary(sess.run(merged_summary, feed_dict={model.loss: loss_value, sentences_placeholder: all_sentences}), global_step=train_step)
    with open(config.summary_path + model_folder + '/sentences.txt', 'a+') as file:
        file.write('\n-------------------------------------------------------------\n')
        file.write(str(train_step) + '\n')
        file.write('-------------------------------------------------------------\n\n')
        for i, sentence in enumerate(all_sentences):
            file.write(sentence + ' | ' + str(all_probs[i]) + '\n')
    # Save model
    save_path = saver.save(sess, checkpoint_path + '/' + 'model', global_step=train_step+1)
    print('Model saved at %s' % (save_path))




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
    # train(config, global_step=5000, model_folder='book_EN_grimms_fairy_tails_20171126-1622/')