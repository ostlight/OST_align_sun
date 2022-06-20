#! /usr/bin/python3
# -*- coding: utf-8 -*-

from glob import glob

from time import time

import sys

import planetary_system_stacker as pss
sys.path.append(pss.__path__[0])

from rank_frames import RankFrames
from configuration import Configuration
from exceptions import Error
from frames import Frames

#names = glob('test_ranking/*.fit')
names = glob('out_2022-05-05-1441_7-1-CapObj/cut/*.fit')
ftype = 'image'
print(names)

# Get configuration parameters.
configuration = Configuration()
configuration.initialize_configuration(read_from_file=False)
try:
    frames = Frames(configuration, names, type=ftype)
    print("Number of images read: " + str(frames.number))
    print("Image shape: " + str(frames.shape))
except Error as e:
    print("Error: " + e.message)
    exit()

# Rank the frames by their overall local contrast.
start = time()
rank_frames = RankFrames(frames, configuration)
rank_frames.frame_score()
end = time()
print('Elapsed time in ranking all frames: {}'.format(end - start))

print(rank_frames.quality_sorted_indices_original)

