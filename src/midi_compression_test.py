from midi_utils import *


filename = "../midis/help.mid"

prep = MidiPreprocessor()

real_notes = prep.midi_to_real_notes(filename)
simple_notes = []

for note in real_notes:
    simple_notes.append(note.to_simple_note())


time_series = prep.simple_notes_to_time_series(simple_note_list=simple_notes)

new_simple_notes = prep.time_series_to_simple_note_list(time_series)
# new_simple_notes = simple_notes
new_real_notes = []

for note in new_simple_notes:
    new_real_notes.append(note.to_real_note())

prep.msgs_to_midi(new_real_notes, "../midis/new_no.mid")

