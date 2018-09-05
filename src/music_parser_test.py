from music21 import *
import numpy as np
from music_parser import *

parser = MusicParser()

MIDI_PATH = '../midis/'

score = converter.parse(r'' + '../midis/alb_esp1_format0.mid')

note_data = parser.score_to_note_data(score)

min = parser.get_smallest_note_value(note_data)

max = parser.get_largest_note_value(note_data)

time_series = parser.note_data_to_time_series(note_data, min, max-min)

print(time_series)

note_data2 = parser.time_series_to_midi(time_series, min, MIDI_PATH)

print(time_series)
