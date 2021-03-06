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

#   Output formal
out_format = ".tiff"

#   Reference image
ref_id = 0


###
#   Extend or cut image -> The images can be either cut to the common
#                          field of view or extended to a field of view
#                          in which all images fit
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
bool_heavy = True

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
#plot_mask = True
plot_mask = False

#   Plot cut images
plot_cut  = True
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

############################################################################
####                               Main                                 ####
############################################################################

#   Check if output directory exists
sys.stdout.write("\rCheck directories...\n")
path_in  = checks.check_pathlib_Path(path_in)
checks.check_out(path_out)

#   Path to the trimmed images
trim_path = Path(Path(path_out) / 'cut')
trim_path.mkdir(exist_ok=True)

#   Make file list
sys.stdout.write("\rRead images...\n")
fileList, nfiles = aux.mkfilelist(
    path_in,
    formats=formats,
    addpath=True,
    sort=True,
    )

#   Read images
im = imread_collection(fileList)

#   Image shape
image_shape = im[0].shape
ny = image_shape[0]
nx = image_shape[1]
nz = image_shape[2]

#   Bit depth
if im[0].dtype == 'uint8':
    bit_depth = 8
elif im[0].dtype == 'uint16':
    bit_depth = 16
else:
    print('Caution: Bit depth could not be determined.')
    print('Data type is ', im[0].dtype)
    print('Bit depth set to 8')
    bit_depth = 8


###
#   Calculate image mask -> the masked area will not be considered,
#                           while calculating the shifts
mask = np.ones((ny,nx), dtype=bool)
if bool_mask:
    sys.stdout.write("\rCalculate image mask...\n")
    if upper_right:
        mask_points.append([0,nx])
    if lower_right:
        mask_points.append([ny,nx])
    if lower_left:
        mask_points.append([ny,0])

    #   Calculate mask from the specified points
    mask = np.invert(polygon2mask((ny,nx), mask_points))


###
#   Calculate shifts
#
#   Prepare an array for the shifts
shifts = np.zeros((2,nfiles), dtype=int)

#   Loop over number of files
sys.stdout.write("\rCalculate shifts...\n")
for i in range(0, nfiles):
    #   Write current status
    id_c = i+1
    sys.stdout.write("\rImage %i/%i" % (id_c, nfiles))
    sys.stdout.flush()

    #   "Normalize" image & calculate heavyside function for the image
    if bool_heavy:
        reff = np.heaviside(
            im[ref_id][:,:,0]/2**(bit_depth)-offset/2**(bit_depth),
            0.03,
            )
        test = np.heaviside(
            im[i][:,:,0]/2**(bit_depth)-offset/2**(bit_depth),
            0.03,
            )
    else:
        reff = im[ref_id][:,:,0]
        test = im[i][:,:,0]

    #   Plot reference image and image mask
    if i == ref_id and plot_mask:
        fig = plt.figure(figsize=(12, 7))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

        ax1.imshow(reff, cmap='gray')
        ax1.set_axis_off()
        ax1.set_title('Reference image')

        ax2.imshow(mask, cmap='gray')
        ax2.set_axis_off()
        ax2.set_title('Image mask')

        plt.show()

    #   Calculate shifts
    shifts[:,i] = phase_cross_correlation(reff, test, reference_mask=mask)

sys.stdout.write("\n")

#   Ensure shifts are Integer and that they have the right sign
shifts = shifts.astype(int)*-1

#   Calculate maximum, minimum, and delta shifts
minx, maxx, miny, maxy, deltax, deltay = aux.cal_delshifts(
    shifts,
    pythonFORMAT=True,
    )


###
#   Cut pictures and add them
#
for i in range(0,nfiles):
    #   Write status to console
    id_c = i+1
    sys.stdout.write("\rApply shift to image %i/%i" % (id_c, nfiles))
    sys.stdout.flush()

    #   Distinguish between image extension and image cutting
    if mode == 'cut':
        #   Determine index positions to cut
        xs, xe, ys, ye =  aux.shiftsTOindex(
            shifts[:,i],
            minx,
            miny,
            nx,
            ny,
            deltax,
            deltay,
            pythonFORMAT=True,
            )

        #   Actual image cutting
        img_i = im[i][ys+ys_cut:ye-ye_cut, xs+xs_cut:xe-xe_cut]

    elif mode == 'extend':
        #   Define larger image array
        img_i = np.zeros(
            (ny+deltay, nx+deltax, nz),
            dtype='uint'+str(bit_depth)
            )

        #   Set Gamma channel if present
        if nz == 4:
            img_i[:,:,3] = 2**bit_depth-1

        #   Calculate start indexes for the larger image array
        ys =  maxy - shifts[0,i]
        xs =  maxx - shifts[1,i]

        #   Actual image manipulation
        img_i[ys:ys+ny, xs:xs+nx] = im[i]

    else:
        print('Error: Mode not known. Check variable "mode".')
        sys.exit()

    #   Set reference image
    if i == ref_id:
        img_ref = img_i

    #   Plot reference and offset image
    if i == id_img and plot_cut and ref_id <= id_img:
        fig = plt.figure(figsize=(12, 7))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

        ax1.imshow(img_ref, cmap=None)
        ax1.set_axis_off()
        ax1.set_title('Reference image')

        ax2.imshow(img_i)
        ax2.set_axis_off()
        ax2.set_title('Offset image')

        plt.show()

    #   Write image
    #new_name = 'shift_'+os.path.basename(fileList[i]).split('.')[0]
    new_name = os.path.basename(fileList[i]).split('.')[0]
    if out_format in [".tiff", ".TIFF"]:
        imsave(trim_path / str(new_name+'.tiff'), img_i[:,:,0:4])
    elif out_format in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        imsave(trim_path / str(new_name+'.jpg'), img_i[:,:,0:3])
    else:
        print('Error: Output format not known :-(')


sys.stdout.write("\n")







