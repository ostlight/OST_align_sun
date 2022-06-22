#! /usr/bin/python3
# -*- coding: utf-8 -*-

import sys

from glob import glob

from time import time

import matplotlib.pyplot as plt

from numpy import array, full

import planetary_system_stacker as pss
sys.path.append(pss.__path__[0])

from rank_frames import RankFrames
from configuration import Configuration
from exceptions import Error
from frames import Frames

names = glob('test_ranking/*.fit')
#names = glob('out_2022-05-05-1441_7-1-CapObj/cut/*.fit')
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

#print(rank_frames.quality_sorted_indices_original)
print(rank_frames.quality_sorted_indices)

for rank, index in enumerate(rank_frames.quality_sorted_indices):
    frame_quality = rank_frames.frame_ranks[index]
    print("Rank: " + str(rank) + ", Frame no. " + str(index) + ", quality: " + str(frame_quality))
for index, frame_quality in enumerate(rank_frames.frame_ranks):
    rank = rank_frames.quality_sorted_indices.index(index)
    print("Frame no. " + str(index) + ", Rank: " + str(rank) + ", quality: " +
            str(frame_quality))

print("")
num_frames = len(rank_frames.frame_ranks)
frame_percent = 10
num_frames_stacked = max(1, round(num_frames*frame_percent/100.))
print("Percent of frames to be stacked: ", str(frame_percent), ", numnber: "
        + str(num_frames_stacked))
quality_cutoff = rank_frames.frame_ranks[rank_frames.quality_sorted_indices[num_frames_stacked]]
print("Quality cutoff: ", str(quality_cutoff))

plt.switch_backend('TkAgg')
# Plot the frame qualities in chronological order.
ax1 = plt.subplot(211)

x = array(rank_frames.frame_ranks)
plt.ylabel('Frame number')
plt.gca().invert_yaxis()
y = array(range(num_frames))
x_cutoff = full((num_frames,), quality_cutoff)
plt.xlabel('Quality')
line1, = plt.plot(x, y, lw=1)
line2, = plt.plot(x_cutoff, y, lw=1)
index = 37
plt.scatter(x[index], y[index], s=20)
plt.grid(True)

# Plot the frame qualities ordered by value.
ax2 = plt.subplot(212)

x = array([rank_frames.frame_ranks[i] for i in rank_frames.quality_sorted_indices])
plt.ylabel('Frame rank')
plt.gca().invert_yaxis()
y = array(range(num_frames))
y_cutoff = full((num_frames,), num_frames_stacked)
plt.xlabel('Quality')
line3, = plt.plot(x, y, lw=1)
line4, = plt.plot(x, y_cutoff, lw=1)
index = 37
plt.scatter(x[index], y[index], s=20)
plt.grid(True)

plt.show()

number = 1
window = 5
start = time()
best_indices, quality_loss_percent, cog_mean_frame = rank_frames.find_best_frames(number, window)
end = time()
print ("\nIndices of best frames in window of size " + str(window) + " found in " +
        str(end - start) + " seconds: " + str(best_indices) +
        "\nQuality loss as compared to unrestricted selection: " +
        str(quality_loss_percent) + "%\nPosition of mean frame in video time line: " +
        str(cog_mean_frame) + "%")
