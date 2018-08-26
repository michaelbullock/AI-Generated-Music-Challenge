"""
Loads all midi files in a directory, parses them by time step and saves them to pandas data frame.
"""

from midi_utils import *

LOAD_DIR = "../small_midis_format0_ab/"
SAVE_DIR = "../data/"
NUM_TIMESTEPS = 32

preprocessor = MidiPreprocessor()
df = preprocessor.all_midis_to_df(LOAD_DIR, NUM_TIMESTEPS)

save_filename = "{}small_midis_format0_ab_{}ts.pkl".format(SAVE_DIR, NUM_TIMESTEPS)

print("Saving data frame to: {}".format(save_filename))

df.to_pickle(save_filename)

