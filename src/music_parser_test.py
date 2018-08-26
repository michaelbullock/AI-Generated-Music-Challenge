from music21 import *
import numpy as np
from music_parser import *

parser = MusicParser()

# MIDI_PATH = 'C:\Users\micha\Documents\Music Generation Challenge'

score = converter.parse(r'C:\Users\micha\Documents\Music Generation Challenge\testmidi.mid')

note_data = parser.score_to_note_data(score)

min = parser.get_smallest_note_value(note_data)

max = parser.get_largest_note_value(note_data)

time_series = parser.note_data_to_time_series(note_data, min, max-min)

print(time_series)

note_data2 = parser.time_series_to_note_data(time_series, min)

print(time_series)

