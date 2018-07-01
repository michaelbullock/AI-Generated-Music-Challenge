"""
Testing parsing functions to see if we can get resolution down to 1/16th notes
"""

from midi_utils import *


filename = "../midis/schubert_D850_1_format0.mid"

preprocessor = MidiPreprocessor()

real_note_list = preprocessor.midi_to_real_notes(filename)
simple_note_list = []

for note in real_note_list:
    simple_note_list.append(note.to_simple_note())


time_series = preprocessor.simple_notes_to_time_series(simple_note_list)

print()


rec_simple_note_list = preprocessor.time_series_to_simple_note_list(time_series)

rec_real_note_list = []

for note in rec_simple_note_list:
    rec_real_note_list.append(note.to_real_note())


preprocessor.msgs_to_midi(rec_real_note_list, "../midis/rec_schubert.mid")



