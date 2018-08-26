import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    def get_smallest_note_value(self, note_data):
        """takes in NoteData object and returns an integer for the lowest note (midi format)"""
        temp = 300
        for note in note_data.notes:
            if note.isNote:
                if temp > note.pitch.midi:
                    temp = note.pitch.midi
            elif note.isChord:
                for component in note.pitches:
                    if temp > component.midi:
                        temp = component.midi

        return temp

    def get_largest_note_value(self, note_data):
        """takes in NoteData object and returns an integer for the highest note (midi format)"""
        temp = -1
        for note in note_data.notes:
            if note.isNote:
                if temp < note.pitch.midi:
                    temp = note.pitch.midi
            elif note.isChord:
                for component in note.pitches:
                    if temp < component.midi:
                        temp = component.midi

        return temp

    def score_to_note_data(self, music):
        """takes in music21 score stream and outputs notedata object with longest list of notes, chords, and rests"""
        # limiting the score object to parts avoids issues with metadata and staffgroup
        music_parts = music.parts
        data = []
        hold = NoteData()
        # runs through elements of the song and stores into NoteData obj
        for element in music_parts.elements:
            for thisNote in element.getElementsByClass(['Note', 'Chord']):
                if thisNote.duration.quarterLength > 3:
                    thisNote.duration.quarterLength = 4
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

    def time_series_to_note_data(self, time_series, min_note):
        s1 = stream.Stream()
        note_data = NoteData()
        for my_offset in range(len(time_series)):
            for index in range(len(time_series[my_offset])):
                if time_series[my_offset][index]:
                    my_type = index % 7
                    if my_type < 5:
                        my_pitch = (index / 7) + min_note
                        newNote = note.Note('C')
                        newNote.pitch.midi = my_pitch
                        s1.append(newNote)
                        newNote.setOffsetBySite(s1, my_offset / 4.0)
                        if my_type == 0:
                            newNote.duration.quarterLength = 4
                        elif my_type == 1:
                            newNote.duration.quarterLength = 2
                        elif my_type == 2:
                            newNote.duration.quarterLength = 1
                        elif my_type == 3:
                            newNote.duration.quarterLength = 0.5
                        elif my_type == 4:
                            newNote.duration.quarterLength = 0.25
                        note_data.notes.append(newNote)
        return note_data
