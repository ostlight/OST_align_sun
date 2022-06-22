#! /usr/bin/python3
# -*- coding: utf-8 -*-

'''
    Make video from individual images
'''

############################################################################
####           Configuration: modify the file in this section           ####
############################################################################

#   Path to the images
# path_in  = 'out_2022-05-05-1441_7-1-CapObj/postpro/'
# path_in  = 'out_2022-05-05-1450_7-1-CapObj/postpro/'
# path_in  = 'out_2022-05-05-1456_5-1-CapObj/postpro/'
# path_in  = 'out_2022-05-05-1506_9-1-CapObj/postpro/'
# path_in  = 'out_2022-05-05-1515_0-1-CapObj/postpro/'
path_in  = 'out_multi/postpro/'

#   Output directory
# path_out = 'out_2022-05-05-1441_7-1-CapObj/'
path_out = '.'

#   Allowed input file formats
#formats = [".jpg", ".jpeg", ".JPG", ".JPEG"]
formats = [".tiff"]

###
#   Video options
#
#   Make a video (True or False)?
mk_video = True
#   Video name
# video_name = '2022-05-05-1450_7-1-CapObj'
# video_name = '2022-05-05-1456_5-1-CapObj_cut_v4'
# video_name = '2022-05-05-1506_9-1-CapObj_v4'
# video_name = '2022-05-05-1515_0-1-CapObj_v4'
# video_name = 'multi_v4'
video_name = 'multi_best_v4'
#   Video annotation
video_annotation = ''
#   Frames per second
fps = 20
fps = 60
#fps = 120

############################################################################
####                            Libraries                               ####
############################################################################

import sys
import os

import warnings
warnings.filterwarnings('ignore')

import cv2

import checks
import aux

############################################################################
####                               Main                                 ####
############################################################################

#   Check directories
sys.stdout.write("\rCheck directories...\n")
path_in  = checks.check_pathlib_Path(path_in)
checks.check_out(path_out)

#   Make file list
fileList, nfiles = aux.mkfilelist(path_in, formats=formats, addpath=True)

#   Read images
sys.stdout.write("\rRead images...")
sys.stdout.write("\n")
img_list = []
for file in fileList:
   img_list.append(cv2.imread(file))

###
#   Write video
#
sys.stdout.write("\rWrite video...")
sys.stdout.write("\n")

aux.write_video(
    os.path.join(path_out, video_name+'.webm'),
    img_list,
    video_annotation,
    fps,
    depth=8,
    )

