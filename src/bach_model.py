import tensorflow as tf
import numpy as np
import datetime
from data_parsing import *
import pickle

def get_time():
    return datetime.datetime.now().strftime("%m-%d--%H-%M")


def get_batch(songs_parsed, batch_size, num_timesteps):
    # create test data
    # loop = np.array([
    #     [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    #     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
    # ])
    # loop_10 = np.concatenate([loop for x in range(10)], axis=0)

    song = songs_parsed[np.random.randint(0, len(songs_parsed))]

    # create lists to hold data, 0-dimension is batch
    X_batch = []
    y_batch = []

    # iterate for batch size
    for x in range(batch_size):
        # pick random starting index
        random_starting_index = np.random.randint(0, len(song) - num_timesteps)

        # y data is X shifted forward 1 timestep
        X = song[random_starting_index:random_starting_index + num_timesteps]
        y = song[random_starting_index + 1:random_starting_index + num_timesteps + 1]

        # append batch list
        X_batch.append(X)
        y_batch.append(y)

    X_batch = np.array(X_batch).reshape(batch_size, num_timesteps, 7*80)
    y_batch = np.array(y_batch).reshape(batch_size, num_timesteps, 7*80)

    return X_batch, y_batch


def build_graph(ONEHOT_LENGTH, NUM_TIMESTEPS, NUM_NOTES, LEARNING_RATE, NETWORK_LAYERS):
    # vanilla initializer
    init = tf.global_variables_initializer()

    # input for X and y data
    notes_in_placeholder = tf.placeholder(tf.float32, [None, NUM_TIMESTEPS, NUM_NOTES * ONEHOT_LENGTH])
    notes_out_placeholder = tf.placeholder(tf.float32, [None, NUM_TIMESTEPS, NUM_NOTES * ONEHOT_LENGTH])

    cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n, activation=tf.nn.tanh) for n in NETWORK_LAYERS]
    stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)

    # wrap stacked lstm into dynamic_rnn wrapper
    lstm_output, lstm_states = tf.nn.dynamic_rnn(stacked_rnn_cell, notes_in_placeholder, dtype=tf.float32)

    # print lstm output
    # lstm_output_p = tf.Print(lstm_output, [lstm_output], summarize=100)

    # split_one_hots = tf.split(input_placeholder, one_hot_template, -1)
    split_one_hots = tf.split(lstm_output, NUM_NOTES, -1)
    # split_one_hots_p = tf.Print(split_one_hots, [split_one_hots], summarize=100)

    # stack em up
    split_one_hots_as_list = [split_one_hots[index] for index in range(NUM_NOTES)]
    stacked_one_hots = tf.stack(split_one_hots_as_list, axis=-2)
    # stacked_one_hots_p = tf.Print(stacked_one_hots, [stacked_one_hots], summarize=100)

    # split and stack the y data to match stacked_one_hots
    notes_out_tensor = tf.convert_to_tensor(notes_out_placeholder)
    notes_out_split = tf.split(notes_out_tensor, NUM_NOTES, -1)
    notes_out_stacked = tf.stack(notes_out_split, axis=-2)

    # softmax with cross entropy is used for calculating loss during training
    softmax_notes_with_CE = tf.nn.softmax_cross_entropy_with_logits(labels=notes_out_stacked, logits=stacked_one_hots)

    # softmax output is used during
    softmax_notes_output = tf.nn.softmax(stacked_one_hots)

    # average output of softmax with cross entropy, optimize with Adam
    loss = tf.reduce_mean(softmax_notes_with_CE)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train = optimizer.minimize(loss)

    return init, train, loss, notes_in_placeholder, notes_out_placeholder, softmax_notes_output


def training_test():
    # network hyper parameters
    NUM_NOTES = 80
    ONEHOT_LENGTH = 7  # whole, half, half-rest, null
    NUM_TIMESTEPS = 5
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NETWORK_LAYERS = [7*80, 7*80]

    NUM_SAMPLES_TO_TRAIN = 400
    SAVE_EVERY = 10000

    PARSED_SONGS = "../data/4scales_parsed.pkl"
    with open(PARSED_SONGS, 'rb') as f:
        songs_parsed = pickle.load(f)

    # build graph
    init, train, loss, notes_in_placeholder, notes_out_placeholder, softmax_notes_output = \
        build_graph(ONEHOT_LENGTH, NUM_TIMESTEPS, NUM_NOTES, LEARNING_RATE, NETWORK_LAYERS)

    # for saving the model
    saver = tf.train.Saver(max_to_keep=100)

    # start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # create loss summary
        log_filename = "../logs/training_test_" + get_time()
        print("saving loss to: " + log_filename)

        writer = tf.summary.FileWriter(log_filename, sess.graph)
        loss_summary = tf.summary.scalar('Loss', loss)

        samples_trained = 0
        while samples_trained < NUM_SAMPLES_TO_TRAIN:
            # begin training routine
            if samples_trained % 10*BATCH_SIZE == 0:
                print("Trained " + str(samples_trained) + " samples")

            samples_trained += BATCH_SIZE

            X, y = get_batch(songs_parsed, BATCH_SIZE, NUM_TIMESTEPS)

            # run graph and get batch loss
            batch_loss, note_predictions, _ = sess.run([loss_summary, softmax_notes_output, train],
                                                       feed_dict={notes_in_placeholder: X, notes_out_placeholder: y})

            # add summary to tensorboard logs
            writer.add_summary(batch_loss, samples_trained)

            # save model at increments
            if samples_trained % SAVE_EVERY == 0:
                variables_save_file = "./models/training_test_" + get_time()
                print("saving model to: " + variables_save_file)
                saver.save(sess, variables_save_file)

        writer.close()

training_test()

