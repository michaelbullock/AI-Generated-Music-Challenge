from basic_music_model import *
import pandas as pd
import tensorflow as tf
from midi_utils import *

MODEL_NAME = "four_schuberts_16ts_07-03--18-11"
GRAPH_NAME = "four_schuberts_16ts_07-03--18-11"


# create starting time step

simple_notes = []

note = SimpleNoteMessage(66, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(70, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(73, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(66, -1, 16)
simple_notes.append(note)
note = SimpleNoteMessage(70, -1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(73, -1, 0)
simple_notes.append(note)

prep = MidiPreprocessor()
time_series_data = prep.simple_notes_to_time_series(simple_notes)

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

tmp = Model(df_file, num_keys, num_timesteps, num_layers, num_neurons_inlayer, learning_rate, batch_size)

model = tmp.load(MODEL_NAME)

tf.reset_default_graph()
generated_time_series = model.generate_music(GRAPH_NAME, time_series_data, 500)

new_simple_notes = prep.time_series_to_simple_note_list(generated_time_series)

new_real_notes = []

for note in new_simple_notes:
    new_real_notes.append(note.to_real_note())

prep.msgs_to_midi(new_real_notes, "../midis/generated.mid")

