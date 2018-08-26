# from midi_utils import *
import pandas as pd
# from basic_music_model import *
import numpy as np
#
# # model parameters
# num_keys = 128
# num_timesteps = 16
# num_layers = 2
# num_neurons_inlayer = 100
# learning_rate = 0.001
# batch_size = 3
#
# # training parameters
# num_samples_to_train = 2000
# save_every = 500
#
# # load data frame
# df_file = "four_schuberts_16ts"
# df = pd.read_pickle("../data/" + df_file + ".pkl")
#
#
#
# # Model(self, df_file, num_keys, num_timesteps, num_layers, num_neurons_inlayer, learning_rate, batch_size):
# model = Model(df_file, num_keys, num_timesteps, num_layers, num_neurons_inlayer, learning_rate, batch_size)
#
#
# # train_model(self, df,  num_samples_to_train, save_every=10000, graph_name=""):
# model.train_model(df, num_samples_to_train, save_every=save_every)
#


"""
batch_size = 1
num_timestep = 16
num_keys = 128


for df_index in range(2):
    df_sample = df.sample(n=3)

    vals = df_sample.values

    x_batch = []
    y_batch = []
    for batch_index in range(len(df_sample.values[:, 0])):
        x_batch.append(np.array(df_sample.values[batch_index, 0]).reshape((batch_size, num_timestep, num_keys)))
        y_batch.append(np.array(df_sample.values[batch_index, 1]).reshape((batch_size, num_timestep, num_keys)))


    print(x)
    print(y)

    print("-----------------")

"""

l = np.array([0, 1, 2, 3])
s = np.array([0, 1])
print(l)

l[2:] = s
print(l)

s[0] = 1111
print(l)

