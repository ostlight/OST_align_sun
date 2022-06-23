#! /usr/bin/python3
# -*- coding: utf-8 -*-

'''
    Load images of specific formats and calculate shifts to a reference
    image
'''


############################################################################
####           Configuration: modify the file in this section           ####
############################################################################

#   Path to the images
#path_in  = '../se_2021_clean_imgs/'
#path_in  = './examples/'
path_in = 'out_test_ser/video_imgs/'

#   Output directory
#path_out = 'out_test'
path_out = 'out_test_ser'

#   Allowed input file formats
formats = [".tiff", ".TIFF"]

#   Output format
out_format = ".tiff"

#   ID of the reference image
ref_id = 0


###
#   Extend or cut image -> The images can be either cut to the common
#                          field of view or extended to a field of view
#                          in which all images fit
#                       -> Limitation: ``extend`` is currently not available
#                                      for FITS files
#
mode = 'extend'
mode = 'cut'


###
#   Image mask -> used to mask image areas that could spoil the
#                 cross correlation such as the moon during a
#                 solar eclipse
#
#   Mask image?
bool_mask = True
bool_mask = False

#   Points to define the mask -> the area enclosed by the points
#                                will be masked
mask_points = [[0, 0]]

#   Add a point for the upper right corner
upper_right = True
#   Add a point for the lower right corner
lower_right = True
#   Add a point for the lower left corner
lower_left  = False


###
#   Apply a heavyside function to the image
#
bool_heavy = False
#bool_heavy = True

#   Background offset
offset = 0


###
#   Additional cuts to the images
#
#   Upper edge
ys_cut = 0
#   Lower edge
ye_cut = 0
#   Left edge
xs_cut = 0
#   Right edge
xe_cut = 0


###
#   Plot options
#
#   Plot the image mask and reference image
plot_mask = True
plot_mask = False

#   Plot cut images
plot_cut  = True
#plot_cut  = False
#   ID of the image to plot
id_img    = 10


############################################################################
####                            Libraries                               ####
############################################################################

import sys
import os

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.draw import polygon2mask
from skimage.io import imread, imread_collection, imsave, imshow

import checks
import aux

import registration


############################################################################
####                               Main                                 ####
############################################################################

if __name__ == '__main__':
    if '.fit' in formats or '.fits' in formats:
        registration.cal_img_shifts_fits(
            path_in,
            path_out,
            formats,
            out_format,
            ref_id=ref_id,
            bool_mask=bool_mask,
            mask_points=mask_points,
            upper_right=upper_right,
            lower_right=lower_right,
            lower_left=lower_left,
            bool_heavy=bool_heavy,
            offset=offset,
            ys_cut=ys_cut,
            ye_cut=ye_cut,
            xs_cut=xs_cut,
            xe_cut=xe_cut,
            plot_mask=plot_mask,
            plot_cut=plot_cut,
            id_img=id_img,
            )
    else:
        registration.cal_img_shifts_normal(
            path_in,
            path_out,
            formats,
            out_format,
            ref_id=ref_id,
            mode=mode,
            bool_mask=bool_mask,
            mask_points=mask_points,
            upper_right=upper_right,
            lower_right=lower_right,
            lower_left=lower_left,
            bool_heavy=bool_heavy,
            offset=offset,
            ys_cut=ys_cut,
            ye_cut=ye_cut,
            xs_cut=xs_cut,
            xe_cut=xe_cut,
            plot_mask=plot_mask,
            plot_cut=plot_cut,
            id_img=id_img,
            )
