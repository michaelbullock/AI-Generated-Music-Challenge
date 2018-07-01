"""
    File: midis_to_df.py
    Author(s): Noah Thurston
    Purpose: Loads all midis files in a directory, parses them by time step
    and saves them to pandas datafram
"""











"""
import pandas as pd
import numpy as np
from midi_utils import *

from os import listdir
from os.path import isfile, join

# hyper parameters
NUM_TIMESTEPS = 3
LOAD_DIR = "../data/"

# read names of all files in LOAD_DIR
all_files = [f for f in listdir(LOAD_DIR) if isfile(join(LOAD_DIR, f))]

# comb out all files that are not .mid
index = 0
while index < len(all_files):
    if all_files[index][-4:] != ".mid":
        del all_files[index]
    index += 1


curr_song = all_files[0]
print("parsing midi: {}".format(curr_song))


labels = ["x", "y"]
df = pd.DataFrame(columns=labels)

for curr_file in all_files:

    for index in range(len(curr_song)-NUM_TIMESTEPS):
        x = curr_song[index:index+NUM_TIMESTEPS]
        y = curr_song[index+1:index+NUM_TIMESTEPS+1]
        df = df.append({labels[0]: x, labels[1]: y}, ignore_index=True)



"""