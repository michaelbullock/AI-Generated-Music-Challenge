import numpy as np
from data_parsing import *
import pickle

# for finding all midi files in a given directorys
from os import listdir
from os.path import isfile, join


def midis_to_parsed_songs():
    music_parser = MusicParser()

    # / Users / Zeus / PycharmProjects / AI - Generated - Music - Challenge / midis / schubert_D850_1_format0.mid
    LOAD_DIR = "../scales/"
    SAVE_DIR = "../data/"
    SAVE_NAME = "4scales_parsed"
    all_files = [f for f in listdir(LOAD_DIR) if isfile(join(LOAD_DIR, f))]

    # comb out all files that are not .mid
    index = 0
    while index < len(all_files):
        if all_files[index][-4:] != ".mid":
            del all_files[index]
        else:
            index += 1

    all_scores_time_series = []

    for score_name in all_files:
        score = converter.parse(LOAD_DIR + score_name)

        note_data = music_parser.score_to_note_data(score)

        time_series = music_parser.note_data_to_time_series(note_data, 40, 80)

        print(time_series)

        all_scores_time_series.append(time_series)

        print()

    print()

    # saved parsed scores as a pickle
    with open(SAVE_DIR + SAVE_NAME + ".pkl", 'wb') as f:
        pickle.dump(all_scores_time_series, f, pickle.HIGHEST_PROTOCOL)


midis_to_parsed_songs()

