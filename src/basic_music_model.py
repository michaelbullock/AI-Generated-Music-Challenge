"""
The most basic music generation model.
N time steps are fed into the model and it predicts N time steps, shifted forward 1 time step.
"""

import numpy as np
import tensorflow as tf
import pandas as pd

import pickle

import datetime


def steep_tanh(x):
    return tf.nn.tanh(10*x)

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

        x_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_keys])
        y_placeholder = tf.placeholder(tf.float32, [None, 1, self.num_keys])

        def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.num_neurons_inlayer, activation=tf.nn.tanh)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [single_cell() for _ in range(self.num_layers)])

        # forget about the output wrapper
        # lstm_with_wrapper = tf.contrib.rnn.OutputProjectionWrapper(stacked_lstm, output_size=self.num_keys)

        lstm_outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x_placeholder, dtype=tf.float32)

        # the output of stacked_lstm is [?, num_timesteps, num_neurons_inlayer], use tf.gather to get only the last time step
        # goal output is [?, 1, num_neurons_inlayer]
        lstm_output_T = tf.transpose(lstm_outputs, [1, 0, 2])
        lstm_last_output = tf.gather(lstm_output_T, int(lstm_output_T.get_shape()[0]) - 1)

        # create a dense layer that connects the last time step output of 100 nodes to 128 keys
        dense_output = tf.layers.dense(inputs=lstm_last_output, units=128, activation=tf.nn.tanh)

        loss = tf.reduce_mean(tf.square(dense_output - y_placeholder))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        return init, train, loss, x_placeholder, y_placeholder, dense_output

    def train_model(self, df, num_samples_to_train, save_every=10000, graph_name=""):
        print("Training started at: " + datetime.datetime.now().strftime("%H:%M:%S"))
        init, train, loss, x_placeholder, y_placeholder, outputs = self.build_graph()

        saver = tf.train.Saver(max_to_keep=100)
        loss_summary = tf.summary.scalar('Loss', loss)

        with tf.Session() as sess:
            sess.run(init)

            if graph_name != "":
                print("Loading graph: %s" % ("../models/" + graph_name))
                saver.restore(sess, "../models/" + graph_name)

            # TensorBoard code
            self.update_save_name()
            filename = "../logs/" + self.current_save_name
            writer = tf.summary.FileWriter(filename, sess.graph)

            learning_rate = tf.placeholder(tf.float32, shape=[])

            curr_sample_counter = 0
            while curr_sample_counter < num_samples_to_train:

                # sample batch from df
                df_sample = df.sample(n=self.batch_size)

                # manipulate data sampled to be a np.array of shape (batch_size, num_timesteps, num_keys)
                x_batch = np.array([])
                y_batch = np.array([])
                for batch_index in range(len(df_sample.values[:, 0])):
                    x_batch = np.append(x_batch, np.array(df_sample.values[batch_index, 0]))
                    y_batch = np.append(y_batch, np.array(df_sample.values[batch_index, 1]))

                x_batch = x_batch.reshape((self.batch_size, self.num_timesteps, self.num_keys))

                # old code when the output was num_timesteps
                # y_batch = y_batch.reshape((self.batch_size, self.num_timesteps, self.num_keys))

                # now the output is only the last timestep
                y_batch = y_batch.reshape((self.batch_size, self.num_timesteps, self.num_keys))[:, -1, :].reshape((self.batch_size, 1, self.num_keys))


                # train network
                loss_result, train_result, sample_loss, dense_output = sess.run([loss_summary, train, loss, outputs],
                                                            feed_dict={x_placeholder: x_batch, y_placeholder: y_batch,
                                                                       learning_rate: self.learning_rate})


                # record loss for TensorBoard
                writer.add_summary(loss_result, curr_sample_counter)
                curr_sample_counter += self.batch_size

                # save model every "save_every" samples
                if curr_sample_counter % save_every == 0:
                    # saving due to the "save_every" condition
                    variables_save_file = "../models/" + self.df_file + "_" + datetime.datetime.now().strftime(
                        "%m-%d--%H-%M")
                    saver.save(sess, variables_save_file)
                    print("\nTrained %d time steps\tTime: %s" % (curr_sample_counter, datetime.datetime.now().strftime("%H:%M:%S")))
                    print("Saved graph to: %s" % variables_save_file)
                    self.save()

            # save model once done training
            print("FINISHED TRAINING, NOW SAVING")
            variables_save_file = "../models/" + self.df_file + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
            print("\nTrained %d time steps\tTime: %s" % (curr_sample_counter, datetime.datetime.now().strftime("%H:%M:%S")))
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

    def generate_music(self, graph_name, starting_time_step_notes, time_series_length):
        """Returns generated time series data"""
        print("generating time series data")

        init, train, loss, x_placeholder, y_placeholder, outputs = self.build_graph()

        saver = tf.train.Saver()

        generated_time_series_data = np.array([])

        generated_time_series_data = np.append(generated_time_series_data, starting_time_step_notes).reshape(-1, 128)


        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, "../models/" + graph_name)
            new = generated_time_series_data[0:0 + self.num_timesteps, :].reshape(1, self.num_timesteps, 128)
            # feed starting notes through model
            # for x in range(len(generated_time_series_data - self.num_timesteps)):
            sess.run(tf.nn.softmax(logits=outputs), feed_dict={x_placeholder: generated_time_series_data[0:self.num_timesteps, :].reshape(1, self.num_timesteps, 128)})

            while len(generated_time_series_data) < time_series_length:
                print(len(generated_time_series_data))
                curr_time_series = generated_time_series_data[-self.num_timesteps:, :].reshape(1, -1, 128)
                pred_time_series = sess.run(outputs, feed_dict={x_placeholder: curr_time_series})

                print()
                random_boundary = np.random.normal(loc=-0.78, scale=0.050)
                # iterate through the last time step of the new notes and round to (1 or -1)
                for note in range(0, 128, 1):

                    # print(random_boundary)
                    if (pred_time_series[0][note] < random_boundary) or (np.sum(generated_time_series_data[-16:, note]) == 16.0):
                        pred_time_series[0][note] = -1
                    else:
                        pred_time_series[0][note] = 1

                generated_time_series_data = np.append(generated_time_series_data, pred_time_series[0, :]).reshape(-1, 128)

        return generated_time_series_data
