import tensorflow as tf
import numpy as np
import datetime
import pickle
from music_parser import *

def get_time():
    return datetime.datetime.now().strftime("%m-%d--%H-%M")


def get_batch(songs_parsed, one_hot_length, num_keys, batch_size, num_timesteps):
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


    # create lists to hold data, 0-dimension is batch
    X_batch = []
    y_batch = []

    # iterate for batch size
    for x in range(batch_size):
        song = songs_parsed[np.random.randint(0, len(songs_parsed))]

        # pick random starting index
        random_starting_index = np.random.randint(0, len(song) - num_timesteps)

        # y data is X shifted forward 1 timestep
        X = song[random_starting_index:random_starting_index + num_timesteps]
        y = song[random_starting_index + 1:random_starting_index + num_timesteps + 1]

        # append batch list
        X_batch.append(X)
        y_batch.append(y)

    X_batch = np.array(X_batch).reshape(batch_size, num_timesteps, one_hot_length*num_keys)
    y_batch = np.array(y_batch).reshape(batch_size, num_timesteps, one_hot_length*num_keys)

    return X_batch, y_batch

### WIP function for adding time indices to data
# def get_batch_with_time_index(songs_parsed, time_index_dict, time_index, one_hot_length, num_keys, batch_size, num_timesteps):
#     # create lists to hold data, 0-dimension is batch
#     X_batch = []
#     y_batch = []
#
#     time_index_dict_length = len(time_index_dict)
#
#     # iterate for batch size
#     for x in range(batch_size):
#         song = songs_parsed[np.random.randint(0, len(songs_parsed))]
#
#         # pick random starting index
#         random_starting_index = np.random.randint(0, len(song) - num_timesteps)
#
#         X_and_y = song[random_starting_index:random_starting_index + num_timesteps + 1]
#
#         # y data is X shifted forward 1 timestep
#         # X = song[random_starting_index:random_starting_index + num_timesteps]
#         for Xy_time_index in range(len(X_and_y)):
#             X_and_y[time_index] = np.append(X_and_y[time_index], time_index_dict[(Xy_time_index + time_index) % time_index_dict_length])
#
#         X = X_and_y[:-1]
#         y = X_and_y[1:]
#
#         # append batch list
#         X_batch.append(X)
#         y_batch.append(y)
#
#     # add the time index vectors if necessary
#     if time_index_dict != {}:
#         X_batch = np.array(X_batch).reshape(batch_size, num_timesteps, one_hot_length*num_keys + time_index_dict_length)
#         y_batch = np.array(y_batch).reshape(batch_size, num_timesteps, one_hot_length*num_keys + time_index_dict_length)
#     else:
#         X_batch = np.array(X_batch).reshape(batch_size, num_timesteps, one_hot_length*num_keys)
#         y_batch = np.array(y_batch).reshape(batch_size, num_timesteps, one_hot_length*num_keys)
#
#     return X_batch, y_batch


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



# training with real parsed music data
def train(PARSED_SONGS_PKL, MODEL_SAVE_DIR):
    # network hyper parameters
    NUM_NOTES = 60
    ONEHOT_LENGTH = 7
    NUM_TIMESTEPS = 64
    BATCH_SIZE = 5
    LEARNING_RATE = 0.001
    NETWORK_LAYERS = [200, 200, ONEHOT_LENGTH*NUM_NOTES]

    NUM_SAMPLES_TO_TRAIN = 10*1000000
    SAVE_EVERY = 50000


    ### Needed if adding time index vectors
    # if time_index_length != 0:
    #     # creates dictionary of one-hot vectors:
    #     # 0: [1, 0, 0, ...]
    #     # 1: [0, 1, 0, ...]
    #     # 2: [0, 0, 1, ...]
    #     time_index_dict = {i:[(0 if j != i else 1) for j in range(time_index_length)] for i in range(time_index_length)}
    # else:
    #     time_index_dict = {}

    # load parsed songs
    with open(PARSED_SONGS_PKL, 'rb') as f:
        songs_parsed = pickle.load(f)

    # build graph
    init, train, loss, notes_in_placeholder, notes_out_placeholder, softmax_notes_output = \
        build_graph(ONEHOT_LENGTH, NUM_TIMESTEPS, NUM_NOTES, LEARNING_RATE, NETWORK_LAYERS)

    # for saving the model
    saver = tf.train.Saver(max_to_keep=100)

    # start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Starting training at: " + get_time())

        # create loss summary
        log_filename = "../logs/test_more_bigger_layers_" + get_time()
        print("saving loss to: " + log_filename)

        writer = tf.summary.FileWriter(log_filename, sess.graph)
        loss_summary = tf.summary.scalar('Loss', loss)

        samples_trained = 0
        while samples_trained < NUM_SAMPLES_TO_TRAIN:
            # begin training routine
            if (samples_trained % (SAVE_EVERY/10)) < BATCH_SIZE:
                print("Trained " + str(samples_trained) + " samples: " + get_time())


            X, y = get_batch(songs_parsed, ONEHOT_LENGTH, NUM_NOTES, BATCH_SIZE, NUM_TIMESTEPS)

            # run graph and get batch loss
            batch_loss, note_predictions, _ = sess.run([loss_summary, softmax_notes_output, train],
                                                       feed_dict={notes_in_placeholder: X, notes_out_placeholder: y})

            # add summary to tensorboard logs
            writer.add_summary(batch_loss, samples_trained)

            # save model at increments
            if samples_trained % SAVE_EVERY == 0:
                variables_save_file = MODEL_SAVE_DIR + "test_more_bigger_layers_" + get_time()
                print("saving model to: " + variables_save_file)
                saver.save(sess, variables_save_file)

            samples_trained += BATCH_SIZE

        writer.close()


def generate(model_path_and_name, TEMP=0.08):
    # network hyper parameters
    NUM_NOTES = 60
    ONEHOT_LENGTH = 7
    NUM_TIMESTEPS = 64
    BATCH_SIZE = 5
    LEARNING_RATE = 0.001
    NETWORK_LAYERS = [200, 200, ONEHOT_LENGTH*NUM_NOTES]


    TIMESTEPS_TO_SAMPLE = 250

    # load parsed songs to get some starter
    PARSED_SONGS_PKL = "../data/7songs.pkl"
    with open(PARSED_SONGS_PKL, 'rb') as f:
        songs_parsed = pickle.load(f)

    # build graph
    init, train, loss, notes_in_placeholder, notes_out_placeholder, softmax_notes_output = \
        build_graph(ONEHOT_LENGTH, NUM_TIMESTEPS, NUM_NOTES, LEARNING_RATE, NETWORK_LAYERS)

    # for loading the model
    saver = tf.train.Saver(max_to_keep=100)

    # start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # restore our model
        saver.restore(sess, model_path_and_name)

        parser = MusicParser()

        # use the beginning of a song as the starter
        X, y = get_batch(songs_parsed, ONEHOT_LENGTH, NUM_NOTES, BATCH_SIZE, NUM_TIMESTEPS)

        # the written_song variable will hold the starting notes and predicted notes
        written_song = X[0]

        timesteps_sampled = 0
        while timesteps_sampled < TIMESTEPS_TO_SAMPLE:

            timesteps_sampled += 1

            feed_in = written_song[-NUM_TIMESTEPS:, :]

            # run graph and get batch loss
            note_predictions = sess.run([softmax_notes_output],
                                                       feed_dict={notes_in_placeholder: [feed_in]})

            # grab last timestep
            last_timestep = note_predictions[0][0][-1][:].reshape(-1)

            # sample from timestep
            for index_divided in range(int(len(last_timestep) / 7)):
                starting_index = index_divided * 7
                ending_index = starting_index + 7

                distribution = last_timestep[starting_index:ending_index]

                chosen_index = sample_note_from_softmax(distribution, temp=TEMP) + starting_index

                for index in range(starting_index, ending_index):
                    if index == chosen_index:
                        last_timestep[index] = 1
                    else:
                        last_timestep[index] = 0

            # append sampled last timestep
            written_song = np.append(written_song, last_timestep).reshape(-1, ONEHOT_LENGTH*NUM_NOTES)

    # time_series_to_midi(self, time_series, min_note, filepath):
    parser.time_series_to_midi(written_song[:-1, :], 30, "../gen")


def sample_note_from_softmax(distribution, temp=0):

    if temp != 0:
        exp_sum = sum([np.exp(i/temp) for i in distribution])
        distribution = [np.exp(i/temp)/exp_sum for i in distribution]

    random_num = np.random.uniform(0, 1, 1)
    curr_index = 0
    curr_sum = distribution[curr_index]
    while curr_sum < random_num:
        curr_index += 1
        curr_sum += distribution[curr_index]

    return curr_index



# train(PARSED_SONGS_PKL="../data/50songs.pkl", MODEL_SAVE_DIR="../test_more_bigger_layers/")
generate("../test_more_bigger_layers/test_more_bigger_layers_12-21--08-35", TEMP=0.08)
# MusicParser.midis_to_time_series_pickle("../50_midis/", "../data/50songs")

"""
TO DO:

    - re-rewrite sample function to actually sample
    - sample before feeding back in
    - get good starting notes: just used part of another song
    - clean up code & push (why are there 2 "training" functions?)
    
    - it doesnt know where it is in the bar...
    what if we attached a "counting" one-hot vector at the end of note
    (UPDATE: this is a huge pain with tf.stack b/c you must stack [7, 7, 7, 16] which is illegal)


- search for hyperparameters: num layers, num nodes, sample size / song choices


12/17/18
- where were we? 
    - most recent model: "../fifty_songs_models/data_set_search_10-04--21-42"
    - we were screwing with hyper parameters, like training set, num timesteps, 
    
- we found:
    - more songs the merrier (we had just compiled 50)
    - 64 timesteps looked good
    - idk if we messed with num layers and num nodes, lets test that:
        - currently: [7*60, 100, 7*60]
        - more layers: [7*60, 100, 100, 7*60]
        - more nodes: [7*60, 200, 7*60]
        
    - we were setting the parameter to 10M, but it was mostly finished at 1M: set to 2M hard this time (aint nobody got time for that)
    - train on dataset of 50 songs
    
    decent example from 200 node hidden: 12-19--01-01.mid
    
"""



