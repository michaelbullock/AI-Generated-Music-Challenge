from basic_music_model import *
import pandas as pd
import tensorflow as tf
from midi_utils import *
import datetime
import pickle

# create starting time steps
simple_notes = []

# simple 3 notes starting notes
"""
note = SimpleNoteMessage(66, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(70, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(73, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(66, -1, 8)
simple_notes.append(note)
note = SimpleNoteMessage(70, -1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(66, 1, 8)
simple_notes.append(note)
note = SimpleNoteMessage(70, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(73, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(66, -1, 8)
simple_notes.append(note)
note = SimpleNoteMessage(70, -1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(73, -1, 8)
simple_notes.append(note)
"""

# ascending notes starting notes
note = SimpleNoteMessage(66, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(58, 1, 0)
simple_notes.append(note)

note = SimpleNoteMessage(66, -1, 8)
simple_notes.append(note)
note = SimpleNoteMessage(58, -1, 0)
simple_notes.append(note)

note = SimpleNoteMessage(67, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(59, 1, 0)
simple_notes.append(note)

note = SimpleNoteMessage(67, -1, 8)
simple_notes.append(note)
note = SimpleNoteMessage(59, -1, 0)
simple_notes.append(note)

note = SimpleNoteMessage(68, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(60, 1, 0)
simple_notes.append(note)

note = SimpleNoteMessage(68, -1, 8)
simple_notes.append(note)
note = SimpleNoteMessage(60, -1, 0)
simple_notes.append(note)

note = SimpleNoteMessage(69, 1, 0)
simple_notes.append(note)
note = SimpleNoteMessage(61, 1, 0)
simple_notes.append(note)

note = SimpleNoteMessage(69, -1, 8)
simple_notes.append(note)
note = SimpleNoteMessage(61, -1, 0)
simple_notes.append(note)

# create preprocessor and starting time step data
prep = MidiPreprocessor()
time_series_data = prep.simple_notes_to_time_series(simple_notes)

# load model.pkl and tf graph
MODEL_NAME = "small_midis_format0_ab_32ts_07-19--02-59"
GRAPH_NAME = "small_midis_format0_ab_32ts_07-19--02-59"
NUM_TIMESTEPS_TO_GENERATE = 500

tmp = Model(" ", 128, 32, 2, 100, 0.001, 5)
model = tmp.load(MODEL_NAME)

tf.reset_default_graph()
generated_time_series = model.generate_music(GRAPH_NAME, time_series_data, NUM_TIMESTEPS_TO_GENERATE)

new_simple_notes = prep.time_series_to_simple_note_list(generated_time_series)

new_real_notes = []

for note in new_simple_notes:
    new_real_notes.append(note.to_real_note())

prep.msgs_to_midi(new_real_notes, "../midis/generated_" + datetime.datetime.now().strftime("%m-%d--%H-%M") + ".mid")

