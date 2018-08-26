"""
Training the basic music model
"""

from basic_music_model import *
import pandas as pd

# model parameters
num_keys = 128
num_timesteps = 32
num_layers = 2
num_neurons_inlayer = 25
learning_rate = 0.0001
batch_size = 5

# training parameters
num_samples_to_train = 4000*1000
save_every = 40*1000

# load data frame
df_file = "small_midis_format0_ab_32ts"
df = pd.read_pickle("../data/" + df_file + ".pkl")

model = Model(df_file, num_keys, num_timesteps, num_layers, num_neurons_inlayer, learning_rate, batch_size)

model.train_model(df, num_samples_to_train, save_every=save_every)
    
