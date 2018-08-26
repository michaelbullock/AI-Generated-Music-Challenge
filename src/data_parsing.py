import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from music21 import *


class NoteData:
    def __init__(self):
        self.notes = []

    def __str__(self):
        if self.notes is not None:
            print(self.notes)
        return ""


class MusicParser:

    def __init__(self):
        pass

    def score_to_note_data(self, music):
        """takes in music21 score stream and outputs notedata object with longest list of notes, chords, and rests"""
        # limiting the score object to parts avoids issues with metadata and staffgroup
        music_parts = music.parts
        data = []
        hold = NoteData()
        # runs through elements of the song and stores into NoteData obj
        for element in music_parts.elements:
            # added slur below to make function more universal
            for thisNote in element.getElementsByClass(['Note', 'Chord', 'Rest']):
                hold.notes.append(thisNote)
            data.append(hold)
            hold = NoteData()
        note_data = NoteData()
        for part in data:
            if len(note_data.notes) < len(part.notes):
                note_data = part
        return note_data

    def note_data_to_time_series(self, note_data, min_note, note_range):
        """takes a notedata object and outputs an array of timeseries data"""
        WHOLE = np.array([1, 0, 0, 0, 0, 0, 0], dtype=bool)
        HALF = np.array([0, 1, 0, 0, 0, 0, 0], dtype=bool)
        QUARTER = np.array([0, 0, 1, 0, 0, 0, 0], dtype=bool)
        EIGHTH = np.array([0, 0, 0, 1, 0, 0, 0], dtype=bool)
        SIXTEENTH = np.array([0, 0, 0, 0, 1, 0, 0], dtype=bool)
        REST = np.array([0, 0, 0, 0, 0, 1, 0], dtype=bool)
        NON = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)

        # range*7 gives each note played a one hot vector of 7 and 16*offset is the total length of the piece in 16th notes
        time_series = np.zeros(
            [int(4 * (note_data.notes[-1].offset + note_data.notes[-1].duration.quarterLength)), note_range * 7],
            dtype=bool)

        # set everything to rest
        for time_step in time_series:
            for i in range(0, note_range):
                time_step[i * 7:i * 7 + 7] = REST

        # iterate through the notes list
        for note in note_data.notes:
            # determine what type the note is
            if note.isNote:
                note_value = note.pitch.midi - min_note
                note_position = int(note.offset * 4)
                note_type = note.duration.type
                if 0 <= note_value < note_range:
                    if note_type == 'whole':
                        one_hot = WHOLE
                        note_duration = 16
                    elif note_type == 'half':
                        one_hot = HALF
                        note_duration = 8
                    elif note_type == 'quarter':
                        one_hot = QUARTER
                        note_duration = 4
                    elif note_type == 'eighth':
                        one_hot = EIGHTH
                        note_duration = 2
                    elif note_type == '16th':
                        one_hot = SIXTEENTH
                        note_duration = 1
                    else:
                        one_hot = np.array([0, 0, 0, 0, 0, 0, 0], dtype=bool)
                        note_duration = 0
                    # section off the duration of note here
                    for i in range(note_position, note_position + note_duration):
                        time_series[i, note_value * 7:note_value * 7 + 7] = NON
                    # add the one hot vector for the note hit
                    time_series[note_position, note_value * 7:note_value * 7 + 7] = one_hot
            elif note.isChord:
                for component in note.pitches:
                    note_value = component.midi
                    note_position = int(note.offset * 4)
                    note_type = note.duration.type
                    if 0 <= note_value < note_range:
                        if note_type == 'whole':
                            one_hot = WHOLE
                            note_duration = 16
                        elif note_type == 'half':
                            one_hot = HALF
                            note_duration = 8
                        elif note_type == 'quarter':
                            one_hot = QUARTER
                            note_duration = 4
                        elif note_type == 'eighth':
                            one_hot = EIGHTH
                            note_duration = 2
                        elif note_type == '16th':
                            one_hot = SIXTEENTH
                            note_duration = 1
                        else:
                            one_hot = np.array([0, 0, 0, 0, 0, 0, 0], dtype=bool)
                            note_duration = 0
                        # section off the duration of note here
                        for i in range(note_position, note_position + note_duration):
                            time_series[i, note_value * 7:note_value * 7 + 7] = NON
                        time_series[note_position, note_value * 7:note_value * 7 + 7] = one_hot
            else:
                print(note)
        return time_series
