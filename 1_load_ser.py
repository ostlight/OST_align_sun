#! /usr/bin/python3
# -*- coding: utf-8 -*-

'''
    Process .ser files
'''

############################################################################
####           Configuration: modify the file in this section           ####
############################################################################

#   Path to video
file_path = 'whitelight_2600_2x2_6.ser'

#   Output directory
path_out = 'out_test_ser/'

#   Image output format
out_format = '.tiff'

############################################################################
####                            Libraries                               ####
############################################################################

import os
import sys

from pathlib import Path, PurePath

import planetary_system_stacker as pss
sys.path.append(pss.__path__[0])
from configuration import Configuration
from frames import Frames
import ser_parser

import cv2

import checks
import aux

from skimage.io import imsave

############################################################################
####                               Main                                 ####
############################################################################

if __name__ == '__main__':
    #   Check directories
    sys.stdout.write("\rCheck directories...\n")
    path_in  = checks.check_pathlib_Path(file_path)
    checks.check_out(path_out)

    #   Path to the postprocess images
    imgs_path = Path(path_out)
    imgs_path = imgs_path / 'video_imgs'
    imgs_path.mkdir(exist_ok=True)
    #imgs_path = PurePath(path_out)
    #imgs_path.joinpath('video_imgs')

    #   Load configuration for the Planetary System Stacker
    configuration = Configuration()
    configuration.initialize_configuration(read_from_file=False)

    # Create the VideoCapture object.
    cap = ser_parser.SERParser(file_path, SER_16bit_shift_correction=True)
    #print(dir(cap))
    #print(type(cap))
    frame_count = cap.frame_count
    #print('cap.frame_count', frame_count)
    digits = len(str(frame_count))
    #print('len(frame_count)', digits)
    #print('cap.frame_number', cap.frame_number)
    #print('cap.color', cap.color)
    print('cap.header', cap.header)
    cap.header['DebayerPattern'] = cv2.COLOR_BayerBG2BGR
    #print('read_trailer', cap.read_trailer())


    #print(cap.sanity_check(file_path))

    #data0 = cap.read_frame_raw(frame_number=0)
    #print(data0)
    #print(data0.shape)
    #print('cap.frame_number', cap.frame_number)
    #print(cap.read_frame_raw())
    #print('cap.frame_number', cap.frame_number)

    #data0 = cap.read_frame(frame_number=0)
    #print(data0)
    #print(data0.shape)
    #print('cap.frame_number', cap.frame_number)
    #print(cap.read_frame())
    #print('cap.frame_number', cap.frame_number)

    data = cap.read_all_frames()
    #print(data)
    #print(len(data))
    #print(data[0].shape)

    for j, img in enumerate(data):
        new_name = aux.get_basename(file_path)+'_'+str(j+1).rjust(digits, '0')
        print(new_name)
        print(imgs_path.name)
        print(new_name+out_format)
        #print(imgs_path.name / new_name+out_format)
        #out_path = Path(imgs_path.name / new_name+out_format)
        out_path = imgs_path / str(new_name+out_format)
        print(out_path)
        imsave(
            #os.path.join(imgs_path.name, new_name)+out_format,
            out_path,
            img,
            )

    ##   Get video as a frames collection
    #frames = Frames(configuration, path_list, type='video')
    #print(dir(frames))
    #print(type(frames))
