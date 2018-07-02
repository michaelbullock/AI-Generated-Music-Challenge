"""
Classes for loading and parsing midi data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mido import Message, MidiFile, MidiTrack

from os import listdir
from os.path import isfile, join
import sys
import datetime


class RealNoteMessage:
    """ stores note-on data pulled from mido messages
    values are unscaled (range of velocity [0,127], but [-1,1] when scaled)
    """
    def __init__(self, note, velocity, time, channel):
        self._note = note
        self._time = time
        self._velocity = velocity
        self._channel = channel

    def to_simple_note(self, flatten_velocity=True):
        """ converts real notes to simple notes
        -velocity==-1 is note off
        -velocity==1 is note on
        -time is rounded to nearest tenth
        -no individual channels
        """
        if flatten_velocity:
            if self._velocity == 0:
                simple_velocity = -1
            else:
                simple_velocity = 1
        else:
            simple_velocity = self._velocity

        # divide time by 10, rounding all numbers up, like math.ceil()
        simple_time = int(self._time/60 + 0.9)

        simple_note = SimpleNoteMessage(self._note, simple_velocity, simple_time)
        return simple_note

    def __str__(self):
        on_off = "off" if (self._velocity == 0 or self._velocity == -1) else "on"
        return "Note {} turns {}, delta {}".format(self._note, on_off, self._time)


class SimpleNoteMessage:
    """ stores simplifed midi data
    -velocity==-1 is note off
    -velocity==1 is note on
    -time is scaled down and rounded up
    -no individual channels
    """
    def __init__(self, note, velocity, time):

        self._note = note
        self._velocity = velocity
        self._time = time

    def __str__(self):
        on_off = "off" if (self._velocity == 0 or self._velocity == -1) else "on"
        return "Note {} turns {}, delta {}".format(self._note, on_off, self._time)

    def __eq__(self, other):
        return (self._note == other._note) & (self._velocity == other._velocity) & (self._time == other._time)

    def __hash__(self):
        return hash(self.to_tuple())

    def to_tuple(self):
        """Needed for hashing a simple note. """
        return (self._note, self._velocity, self._time)

    def to_real_note(self):
        """ converts simple notes to real notes
        -velocity==-1 is note off
        -velocity==1 is note on
        -time is scaled back up (60)
        """
        if self._velocity == -1:
            real_velocity = 0
        elif self._velocity == 1:
            real_velocity = 64
        else:
            real_velocity = self._velocity

        # multiply by 60
        real_time = (self._time * 60)  # - 240

        # channel is assumed to be 0
        real_note = RealNoteMessage(self._note, real_velocity, real_time, 0)
        return real_note


class MidiPreprocessor:
    """Class to hold preprocessing methods. """
    def __init__(self):
        pass

    def simple_notes_to_time_series(self, simple_note_list):
        
        time_series_data = np.array([])
        time_step_notes = np.zeros(128)-1

        curr_msg_index = 0
        zero_note_list = []
        # iterate through all messages
        while curr_msg_index < len(simple_note_list):

            # get time to wait before executing note event
            time_delta = simple_note_list[curr_msg_index]._time

            if time_delta == 0:
                # edit the temporary time_step_notes variable before saving it for the next time_delta
                curr_note = simple_note_list[curr_msg_index]._note
                time_step_notes[curr_note] = simple_note_list[curr_msg_index]._velocity

                for note in zero_note_list:
                    if curr_note == note:
                        # go back and find that note in previous timestep and turn it off
                        # fixes some issues with rhythm errors
                        time_series_data[len(time_series_data)-1][curr_note] = -1
                zero_note_list.append(curr_note)
            else:
                # when the time delta is not zero, write in time_step_notes for each new time step
                while time_delta > 0:
                    time_series_data = np.append(time_series_data, time_step_notes).reshape(-1, 128)
                    time_step_notes = np.copy(time_step_notes)
                    time_delta -= 1
                time_step_notes[simple_note_list[curr_msg_index]._note] = simple_note_list[curr_msg_index]._velocity
                zero_note_list = [simple_note_list[curr_msg_index]._note]
            curr_msg_index += 1


        return time_series_data

    """
    # old simple notes to time series
    def simple_notes_to_time_series(self, simple_note_list):
        
        time_series_data = np.array([])
        time_step_notes = np.zeros(128)-1

        curr_msg_index = 0
        time_delta = 0

        # iterate through all messages
        while curr_msg_index < len(simple_note_list):
            # print("curr_msg_index: {}".format(curr_msg_index))

            # get time to wait before executing note event
            time_delta = simple_note_list[curr_msg_index]._time

            # wait through time without changes, just copy previous notes
            while time_delta > 0:
                time_series_data = np.append(time_series_data, time_step_notes).reshape(-1, 128)
                time_step_notes = np.copy(time_step_notes)
                time_delta -= 1

            # assuming we've waited through all necessary times
            assert time_delta == 0, "ERROR: time_delta should be 0: not {}".format(time_delta)

            while time_delta == 0:
                time_step_notes[simple_note_list[curr_msg_index]._note] = simple_note_list[curr_msg_index]._velocity
                curr_msg_index += 1
                if curr_msg_index < len(simple_note_list):
                    time_delta = simple_note_list[curr_msg_index]._time
                else:
                    break

            time_series_data = np.append(time_series_data, time_step_notes).reshape(-1, 128)
            time_step_notes = np.copy(time_step_notes)

        return time_series_data
    """
    def time_steps_are_same(self, time1, time2):
        """ Checks if all notes in the same time step are the same """
        sets_are_same = True
        for note_index in range(len(time1)):
            if time1[note_index] != time2[note_index]:
                sets_are_same = False
                break
        return sets_are_same

    def time_series_to_simple_note_list(self, time_series_data):
        """Converts time series data to simple messages (necessary for producing midis). """
        # create blank time_step_notes to use for comparison at first
        temp_step_notes = np.zeros(128) - 1
        simple_note_list = []

        time_delta = 0
        # iterate through each time index
        for time_index in range((len(time_series_data)-1)):
            curr_notes = time_series_data[time_index]
            # check if time step are the same so we can increase the time delta
            if self.time_steps_are_same(temp_step_notes, curr_notes):
                time_delta += 1
            # otherwise, log the changes as simple note messages
            else:
                for note_index in range(len(curr_notes)):
                    # if a note has changed
                    if temp_step_notes[note_index] != curr_notes[note_index]:
                        temp_step_notes[note_index] = curr_notes[note_index]
                        # create message with time_delta=0 and append to list
                        # SimpleNoteMessage(note, velocity, time)
                        simple_note_msg = SimpleNoteMessage(note_index, curr_notes[note_index], time_delta)
                        simple_note_list.append(simple_note_msg)
                        time_delta = 0
                time_delta = 1
        return simple_note_list

    """
    # old time series to simple note list 
   def time_series_to_simple_note_list(self, time_series_data):
        # Converts time series data to simple messages (necessary for producing midis). 

        simple_note_list = []

        time_index = 0
        time_delta = 0

        curr_notes = time_series_data[time_index]
        next_notes = time_series_data[time_index+1]

        # create notes from first time step without checking previous time step (doesn't exist)
        for note_index in range(len(time_series_data[0])):
            if time_series_data[0][note_index] != -1:
                simple_note_msg = SimpleNoteMessage(note_index, 1, 0)
                simple_note_list.append(simple_note_msg)

        time_index = 1
        time_delta = 0

        # iterate through all time steps
        while time_index < (len(time_series_data)-1):
            #while time_index < 10:

            # if nothing has changed after a new time step, increase the time_delta counter and current time counter

            # using set and hash function broken (maybe fix later)
            # while set(time_series_data[time_index]) == set(time_series_data[time_index-1]):
            while self.time_steps_are_same(time_series_data[time_index], time_series_data[time_index-1]) and time_index < (len(time_series_data)-1):
                time_delta += 1
                time_index += 1

            # since something changed, find which notes changed and create messages
            for note_index in range(len(time_series_data[time_index])):
                # if a note has changed
                if time_series_data[time_index][note_index] != time_series_data[time_index-1][note_index]:
                    # create message with time_delta=0 and append to list
                    # SimpleNoteMessage(note, velocity, time)
                    simple_note_msg = SimpleNoteMessage(note_index, time_series_data[time_index][note_index], time_delta)
                    simple_note_list.append(simple_note_msg)
                    time_delta = 0

            # after checking all notes, go to next time step
            time_index += 1
            time_delta = 0

        return simple_note_list
    """
    def msgs_to_midi(self, real_note_list, filename):
        """Takes a list of real messages and creates a midi track"""
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        track.append(Message('program_change', program=12, time=0))

        for note in real_note_list:
            track.append(Message('note_on', note=int(note._note), velocity=int(note._velocity), time=int(note._time)))

        mid.save(filename)

    def plot_time_series(self, time_series_data):
        """Plots time series data to show a midi graph. """
        transposed_data = np.transpose(time_series_data)
        plt.imshow(transposed_data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Notes Played vs Time")
        plt.xlabel("Time")
        plt.ylabel("Note")
        plt.show()

    def midi_to_real_notes(self, filename, max_messages=1e+9):
        """Function that reads a midi file and returns a list of real notes. """

        mid = MidiFile(filename)
        piano_tracks = []

        # pick out only track with largest number of messages
        for track in mid.tracks:
            if(len(track) > len(piano_tracks)):
                piano_tracks = track
        # returned list containing all real notes parsed from the first track of the given file
        real_note_list = []
        curr_num_messages = 0
        wait_time = 0

        # iterate through all piano tracks
        for msg in piano_tracks:
            # just in case track is brokenly long
            if curr_num_messages > max_messages:
                    print("ERROR: REACHED MAX MESSAGES: {}".format(max_messages))
                    break
            else:
                curr_num_messages += 1

                # if note on message, record
            if msg.type == "note_on":
                # all_note_on_msgs.append(msg)
                real_note_message = RealNoteMessage(msg.note, msg.velocity, msg.time + wait_time, msg.channel)
                wait_time = 0
                real_note_list.append(real_note_message)

            # if not note message, still check if it has a wait time, record if it does
            elif hasattr(msg, 'time'):
                if (msg.time > 0) and (len(real_note_list)>0):
                    wait_time += int(msg.time)

        return real_note_list

    def midi_to_df(self, df, labels, time_series_data, num_timesteps):
        """Takes time series data of an individual song and adds network input/output data to a df. """
        # not correctly casting to np.int8
        time_series_data = time_series_data.astype(np.int8, copy=False)
        for index in range(len(time_series_data) - num_timesteps):
            x = time_series_data[index:index + num_timesteps]
            y = time_series_data[index + 1:index + num_timesteps + 1]
            df = df.append({labels[0]: x, labels[1]: y}, ignore_index=True)

        print("Data frame is {} MB".format(sys.getsizeof(df)/1e6))
        return df

    def all_midis_to_df(self, load_dir, num_timesteps):
        """ Takes path to target directory and parses all midis in it, returning a dataframe. """

        # read names of all files in LOAD_DIR
        all_files = [f for f in listdir(load_dir) if isfile(join(load_dir, f))]

        # comb out all files that are not .mid
        index = 0
        while index < len(all_files):
            if all_files[index][-4:] != ".mid":
                del all_files[index]
            index += 1

        # create data frame for data
        labels = ["x", "y"]
        df = pd.DataFrame(columns=labels)

        # iterate through all files in directory
        for filename in all_files:
            print("{}, Parsing midi file: {}".format(datetime.datetime.now().strftime("%H:%M:%S"), filename))

            # load list of notes as real notes
            real_note_list = self.midi_to_real_notes(load_dir + filename)

            # convert list of real notes to simple notes
            simple_note_list = []
            for real_note in real_note_list:
                simple_note_list.append(real_note.to_simple_note())

            time_series_data = self.simple_notes_to_time_series(simple_note_list)

            df = self.midi_to_df(df, labels, time_series_data, num_timesteps)

        print("Final data frame is {} MB".format(sys.getsizeof(df)/1e6))

        return df
