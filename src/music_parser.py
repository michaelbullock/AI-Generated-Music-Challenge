import numpy as np
from music21 import *
import pickle
import datetime

# for finding all midi files in a given directory
from os import listdir
from os.path import isfile, join


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

        k = music.analyze('key')
        i = interval.Interval(k.tonic, pitch.Pitch('C'))
        new_music = music.transpose(i)
        music_parts = new_music.parts
        music_parts = music_parts.voicesToParts()
        data = []
        hold = NoteData()
        # runs through elements of the song and stores into NoteData obj
        for element in music_parts.elements:
            for thisNote in element.getElementsByClass(['Note', 'Chord', 'Voice']):
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
                    note_value = component.midi - min_note
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
                        for i in range(note_position, note_position + note_duration):
                            time_series[i, note_value * 7:note_value * 7 + 7] = NON
                        # add the one hot vector for the note hit
                        time_series[note_position, note_value * 7:note_value * 7 + 7] = one_hot
            else:
                print(note)
        return time_series

    def time_series_to_midi(self, time_series, min_note, filepath):
        """creates a midi file at location specified by filepath arg"""
        s1 = stream.Part()
        for my_offset in range(len(time_series)):
            for index in range(len(time_series[my_offset])):
                if time_series[my_offset][index]:
                    my_type = index % 7
                    if my_type < 5:
                        my_pitch = (index // 7) + min_note
                        newNote = note.Note('C')
                        newNote.pitch.midi = my_pitch
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
                        s1.insertIntoNoteOrChord(my_offset / 4.0, newNote)
                        # note_data.notes.append(newNote)
        # parser = MusicParser()
        score = stream.Score()
        score.insert(0, s1)
        # note_data = parser.score_to_note_data(score)
        mf = midi.translate.streamToMidiFile(score)
        midi_fp = filepath + '/' + datetime.datetime.now().strftime("%m-%d--%H-%M") + '.mid'
        mf.open(midi_fp, 'wb')
        mf.write()
        mf.close()
        return


# def midis_to_parsed_songs(LOAD_DIR, SAVE_DIR, SAVE_NAME):
#     music_parser = MusicParser()
#
#     # LOAD_DIR = "../scales/"
#     # SAVE_DIR = "../data/"
#     # SAVE_NAME = "4scales_parsed"
#     all_files = [f for f in listdir(LOAD_DIR) if isfile(join(LOAD_DIR, f))]
#
#     # comb out all files that are not .mid
#     index = 0
#     while index < len(all_files):
#         if all_files[index][-4:] != ".mid":
#             del all_files[index]
#         else:
#             index += 1
#
#     # create list of all parsed songs that will be saved as a pickle
#     all_songs_time_series = []
#
#     # iterate over all midis in the directory
#     for score_name in all_files:
#         score = converter.parse(LOAD_DIR + score_name)
#
#         note_data = music_parser.score_to_note_data(score)
#         time_series = music_parser.note_data_to_time_series(note_data, 40, 80)
#
#         all_songs_time_series.append(time_series)
#
#
#     # saved parsed scores as a pickle
#     with open(SAVE_DIR + SAVE_NAME + ".pkl", 'wb') as f:
#         pickle.dump(all_songs_time_series, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def midis_to_time_series_pickle(PATH_TO_MIDIS, SAVE_PATH_AND_NAME):
        # manually chosen, pick better parameter better
        min = 30
        max = 90

        # create parser
        parser = MusicParser()

        # read names of all files in LOAD_DIR
        all_files = [f for f in listdir(PATH_TO_MIDIS) if isfile(join(PATH_TO_MIDIS, f))]

        # comb out all files that are not .mid
        index = 0
        while index < len(all_files):
            if all_files[index][-4:] != ".mid":
                del all_files[index]
            index += 1

        all_songs_parsed = []

        # iterate through all files in directory
        for filename in all_files:
            print("parsing: " + filename)
            score = converter.parse(r'' + PATH_TO_MIDIS + filename)
            note_data = parser.score_to_note_data(score)
            # min = self.get_smallest_note_value(note_data)
            # max = self.get_largest_note_value(note_data)
            time_series = parser.note_data_to_time_series(note_data, min, max - min)
            all_songs_parsed.append(time_series)

        with open(SAVE_PATH_AND_NAME + ".pkl", 'wb') as f:
            pickle.dump(all_songs_parsed, f, pickle.HIGHEST_PROTOCOL)
