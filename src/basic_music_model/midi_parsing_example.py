"""
Midi parsing in order to validate our preprocessing methods.
i.e. to answer the question "are our preprocessing methods botching the data?"
"""

from midi_utils import *
from mido import MidiFile
import matplotlib.pyplot as plt


mid = MidiFile("../midis/ty_januar.mid")

real_note_list = []
simple_note_list = []
all_msgs = []
all_note_on_msgs = []
wait_time = 0


### need midi to real message conversion, moved to midi_utils.py

print("converting real note list to simple note list")
for real_note in real_note_list:
    simple_note_list.append(real_note.to_simple_note(flatten_velocity=True))

print("creating time series data")
time_series_data = msgs_to_time_series(simple_note_list)

print("plotting time series data")
plot_time_series(time_series_data[:200])

recreated_simple_note_list = time_series_to_msgs(time_series_data)

# going from time series back to msgs back to time series then plot
# recreated_time_series_data = msgs_to_time_series(recreated_simple_note_list)
# plot_time_series(recreated_time_series_data[:200])

# going from simple to real notes
recreated_real_note_list = []
for note in recreated_simple_note_list:
    real_note = note.to_real_note()
    recreated_real_note_list.append(real_note)

NEW_FILENAME = "ty_recreated.mid"

print("creating midi: {}".format(NEW_FILENAME))
msgs_to_midi(recreated_real_note_list, NEW_FILENAME)

