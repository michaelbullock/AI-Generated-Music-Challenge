from music21 import *
import numpy as np
from data_parsing import *

score = converter.parse(r'C:\Users\micha\Documents\Music Generation Project\AI-Generated-Music-Challenge\midis\test.mid')

note_data = MusicParser.score_to_note_data(score)

time_series = MusicParser.note_data_to_time_series(note_data, 40, 40)

print(time_series)

