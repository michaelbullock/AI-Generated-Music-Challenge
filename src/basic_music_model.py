# rnn that can generate sentences

import numpy as np
import tensorflow as tf
import pandas as pd

import pickle

import datetime

class Model(object):
    def __init__(self, df_file, num_keys, num_timesteps, num_layers, num_neurons_inlayer, learning_rate, batch_size):
        self.num_timesteps = num_timesteps
        self.num_layers = num_layers
        self.num_neurons_inlayer = num_neurons_inlayer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_keys = num_keys
        self.df_file = df_file
        self.current_save_name = ""


    def print_model_info(self):
        print("\n")
        print("Model info:")
        print("num_keys: %d" % self.num_keys)
        print("num_timesteps: %d" % self.num_timesteps)
        print("num_layers: %d" % self.num_layers)
        print("num_neurons_inlayer: %d" % self.num_neurons_inlayer)
        print("learning_rate: %.8f" % self.learning_rate)
        print("batch_size: %d" % self.batch_size)
        print("num_keys: %d" % self.num_keys)
        print("df_file: %s" % self.df_file)
        print("current_save_name: %s" % self.current_save_name)
        print("\n")


    def build_graph(self):
        print("Building model")

        X_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_keys])
        y_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_keys])


        def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.num_neurons_inlayer, activation=tf.nn.tanh)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [single_cell() for _ in range(self.num_layers)])

        lstm_with_wrapper = tf.contrib.rnn.OutputProjectionWrapper(
            stacked_lstm,
            output_size=self.num_keys)

        outputs, states = tf.nn.dynamic_rnn(lstm_with_wrapper, X_placeholder, dtype=tf.float32)

        # TO DO: funky stuff, clean up later
        learning_rate = tf.placeholder(tf.float32, shape=[])
        learning_rate = tf.cast(self.learning_rate, tf.float32)

        # loss = tf.reduce_mean(tf.nn.cross_entropy_with_logits(logits=outputs, labels=y_placeholder))
        loss = tf.reduce_mean(tf.square(outputs - y_placeholder))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(loss)

        # sentence_loss_pl = tf.placeholder(tf.float32, [])

        init = tf.global_variables_initializer()

        return init, train, loss, X_placeholder, y_placeholder, outputs

    def train_model(self, df,  num_samples_to_train, save_every=10000, graph_name=""):
        print("Training started at: " + datetime.datetime.now().strftime("%H:%M:%S"))
        init, train, loss, X_placeholder, y_placeholder, outputs = self.build_graph()

        saver = tf.train.Saver(max_to_keep=100)
        loss_summary = tf.summary.scalar('Loss', loss)

        # sentence_loss_list = np.array([])

        # sentence_loss_summary = tf.summary.scalar('Sentence Loss', sentence_loss_pl)


        ### TRAINING MODEL
        with tf.Session() as sess:
            sess.run(init)

            if graph_name != "":
                print("Loading graph: %s" % ("../models/" + graph_name))
                saver.restore(sess, "../models/" + graph_name)

            #TensorBoard code
            #summaryMerged = tf.summary.merge_all()
            self.update_save_name()
            filename = "../logs/" + self.current_save_name
            writer = tf.summary.FileWriter(filename, sess.graph)

            learning_rate = tf.placeholder(tf.float32, shape=[])

            curr_sample_counter = 0

            for curr_sample in range(num_samples_to_train):


                df_sample = df.sample(n=self.batch_size)


                x_batch = np.array(df_sample.values[0, 0]).reshape((self.batch_size, self.num_timesteps, self.num_keys))
                y_batch = np.array(df_sample.values[0, 1]).reshape((self.batch_size, self.num_timesteps, self.num_keys))

                loss_result, train_result, sample_loss = sess.run([loss_summary, train, loss],
                                                            feed_dict={X_placeholder: x_batch, y_placeholder: y_batch,
                                                                       learning_rate: self.learning_rate})

                writer.add_summary(loss_result, curr_sample_counter)
                curr_sample_counter += 1

                if curr_sample_counter % save_every == 0:
                    # saving due to the "save_every" condition
                    variables_save_file = "../models/" + self.df_file + "_" + datetime.datetime.now().strftime(
                        "%m-%d--%H-%M")
                    saver.save(sess, variables_save_file)
                    print("\nTrained %d sentences\tTime: %s" % (curr_sample_counter, datetime.datetime.now().strftime("%H:%M:%S")))
                    print("Saved graph to: %s" % variables_save_file)
                    self.save()

            # save once done training
            print("FINISHED TRAINING, NOW SAVING")
            variables_save_file = "../models/" + self.df_file + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
            print("\nTrained %d sentences\tTime: %s" % (curr_sample_counter, datetime.datetime.now().strftime("%H:%M:%S")))
            print("Saved graph to: %s" % variables_save_file)
            saver.save(sess, variables_save_file)
            writer.close()
            self.save()
        return


    def save(self):
        self.update_save_name()
        print("Saving model: %s" % self.current_save_name)
        # save model as a .pkl

        with open("../models/" + self.current_save_name + '.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        print("Loading model: %s" % filename)
        # load model as a .pkl

        with open("../models/" + filename + ".pkl", 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def update_save_name(self):
        self.current_save_name = self.df_file + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
