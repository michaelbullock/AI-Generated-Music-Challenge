"""
Training the basic music model
"""

from basic_music_model import *
import pandas as pd

# model parameters
num_keys = 128
num_timesteps = 16
num_layers = 2
num_neurons_inlayer = 100
learning_rate = 0.001
batch_size = 5

# training parameters
num_samples_to_train = 2000
save_every = 500

# load data frame
df_file = "four_schuberts_16ts"
df = pd.read_pickle("../data/" + df_file + ".pkl")

model = Model(df_file, num_keys, num_timesteps, num_layers, num_neurons_inlayer, learning_rate, batch_size)

model.train_model(df, num_samples_to_train, save_every=save_every)

