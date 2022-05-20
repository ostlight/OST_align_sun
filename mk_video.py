#! /usr/bin/python3
# -*- coding: utf-8 -*-

'''
    Make video from individual images
'''

############################################################################
####           Configuration: modify the file in this section           ####
############################################################################

#   Path to the images
path_in  = 'out_2022-05-05-1441_7-1-CapObj/cut/'

#   Output directory
path_out = 'out_2022-05-05-1441_7-1-CapObj/'

#   Allowed input file formats
#formats = [".jpg", ".jpeg", ".JPG", ".JPEG"]
formats = [".FIT",".fit",".FITS",".fits"]

#   Upper edge
ys_cut = 300
#   Lower edge
ye_cut = 270
#   Left edge
xs_cut = 530
#   Right edge
xe_cut = 520

###
#   Make RGB from Gray scale? (only for not FITS file formats)
#
mkRGB = True

#   Scaling parameters
#r_scale = 1.0
#g_scale = 0.7
#b_scale = 0.7
#r_scale = 0.85
#g_scale = 0.4
#b_scale = 0.4
#r_scale = 0.85
#g_scale = 0.5
#b_scale = 0.5
r_scale = 0.9
g_scale = 0.5
b_scale = 0.5
#r_scale = 0.9
#g_scale = 0.45
#b_scale = 0.45


###
#   Image postprocessing
#
#   Global histogram equalization
global_histo_equalization = False

#   Local histogram equalizapath_in  = 'tion
local_histo_equalization = False
#   Footprint for local histogram equalization (disk morphology is used)
disk_size = 30

#   Adaptive histogram equalization
adaptive_histo_equalization = False
#   Clip limit for adaptive histogram equalization
clip_limit = 0.03
clip_limit = 0.002

#   log contrast adjustment
log_cont_adjust = False
#   gain for log contrast adjustment
log_gain = 1.

#   Gamma contrast adjustment
gamma_adjust = False
#   Gamma value
gamma = 1.3

#   Contrast stretching
contrast_stretching = True
#   Upper percentile for contrast stretching
upper_percentile = 98.
upper_percentile = 100.
#   Lower percentile for contrast stretching
lower_percentile = 2.
lower_percentile = 0.

#   Define parameters for postprocessing/sharpening
#   (multiple "layers" are possible)
#
#   Parameters
#   ----------
#       radius          : `float`
#           Radius (in pixels) of the Gaussian sharpening kernel.
#       amount          : `float`
#           Amount of sharpening for this layer.
#       bi_fraction     : `float`
#           Fraction of bilateral vs. Gaussian filter (0.: only Gaussian,
#           1.: only bilateral).
#       bi_range        : `float`
#           Luminosity range parameter of bilateral filter
#           (0 <= bi_range <= 255).
#       denoise         : `float`
#           Fraction of Gaussian blur to be applied to this layer
#           (0.: No Gaussian blur, 1.: Full filter application).
#       luminance_only  : `boolean`
#           True, if sharpening is to be applied to the luminance
#           channel only.
#
#   Usage: [radius, amount, bi_fraction, bi_range, denoise, luminance_only]

#   Example:
postprocess_layers = [
    [1.9, 6., 0.5, 20, 0.8, False,],
    [2.4, 6., 0.35, 20, 0.72, False],
    [3.9, 2.0, 0.35, 20, 0.72, False],
    ]

###
#   Video options
#
#   Make a video (True or False)?
mk_video = True
#   Video name
video_name = '2022-05-05-1441_7-1-CapObj'
#   Video annotation
video_annotation = ''
#   Frames per second
fps = 20

############################################################################
####                            Libraries                               ####
############################################################################

import sys
import os

#import tempfile

from pathlib import Path

import numpy as np

import ccdproc as ccdp

import warnings
warnings.filterwarnings('ignore')

from skimage import data
from skimage.io import imsave

from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

import checks
import aux

############################################################################
####                               Main                                 ####
############################################################################

#   Check directories
sys.stdout.write("\rCheck directories...\n")
path_in  = checks.check_pathlib_Path(path_in)
checks.check_out(path_out)

#   Path to the postprocess images
post_path = Path(Path(path_out) / 'postpro')
post_path.mkdir(exist_ok=True)

#   Get file collection
sys.stdout.write("\rRead images...\n")
ifc = ccdp.ImageFileCollection(path_in)

#   Apply filter to the image collection
#   -> This is necessary so that the path to the image directory is added
#      to the file names. This is required for `calculate_image_shifts`.
ifc = ifc.filter(SIMPLE=True)

#   Number of files
nfiles = len(ifc.files)


###
#   Cut pictures and add them
#
#   Loop over and trim all images
i = 0
img_list = []
for img_ccd, fname in ifc.ccds(ccd_kwargs={'unit': 'adu'}, return_fname=True):
    #   Write status to console
    sys.stdout.write("\rApply cuts to images %i/%i" % (i+1, nfiles))
    sys.stdout.flush()
    i += 1

    #   Trim images
    img_out = ccdp.trim_image(img_ccd[ys_cut:-ye_cut, xs_cut:-xe_cut])


    ###
    #   Postprocessing
    #

    #   Global histogram equalize
    if global_histo_equalization:
        img_glob_eq = exposure.equalize_hist(img_out.data)
        img_out.data = img_glob_eq*2**(bit_depth)

    #   Local equalization
    if local_histo_equalization:
        footprint = disk(disk_size)
        img_eq = rank.equalize(img_out.data, footprint)
        img_out.data = img_eq

    #   Adaptive histogram equalization
    if adaptive_histo_equalization:
        img_adapteq = exposure.equalize_adapthist(
            img_out.data,
            clip_limit=clip_limit,
            )
        img_out.data = img_adapteq*2**(bit_depth)

    #   log contrast adjustment
    if log_cont_adjust:
        logarithmic_corrected = exposure.adjust_log(img_out.data, log_gain)
        img_out.data = logarithmic_corrected

    #   Gamma contrast adjustment
    if gamma_adjust:
        gamma_corrected = exposure.adjust_gamma(img_out.data, gamma)
        img_out.data = gamma_corrected

    #   Contrast stretching
    if contrast_stretching:
        plow, pup = np.percentile(
            img_out.data,
            (lower_percentile, upper_percentile),
            )
        img_rescale = exposure.rescale_intensity(
            img_out.data,
            in_range=(plow, pup),
            )
        img_out.data = img_rescale


    ###
    #   Sharp the image
    #
    #   Default layer
    layers = [aux.PostprocLayer(1., 1., 0., 20, 0., False)]

    #   Add user layer
    for layer in postprocess_layers:
        layers.append(aux.PostprocLayer(*layer))

    #   Sharp/prostprocess image
    img_out.data = aux.post_process(img_out.data, layers)


    ###
    #   Prepare array with RGB image
    #
    if mkRGB and out_format not in [".FIT",".fit",".FITS",".fits"]:
        #   Get shape of the trimmed image
        out_shape = img_out.data.shape

        #   Prepare array
        rgb_img = np.zeros(
            (out_shape[0], out_shape[1], 4),
            dtype='uint8',
            )

        #   Scale data, convert data to 8bit range and add data to the array
        rgb_img[:,:,0] = img_out.data * r_scale / 2**(bit_depth) * 255
        rgb_img[:,:,1] = img_out.data * g_scale / 2**(bit_depth) * 255
        rgb_img[:,:,2] = img_out.data * b_scale / 2**(bit_depth) * 255
        rgb_img[:,:,3] = 255


    ###
    #   Write postprocess image
    #
    new_name = os.path.basename(fname).split('.')[0]
    if out_format in [".tiff", ".TIFF"]:
        if mkRGB:
            imsave(
                os.path.join(path_out, 'postpro', new_name)+'.tiff',
                rgb_img,
                )
        else:
            imsave(
                os.path.join(path_out, 'postpro', new_name)+'.tiff',
                img_out.data,
                )
    elif out_format in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        if mkRGB:
            imsave(
                os.path.join(path_out, 'postpro', new_name)+'.jpg',
                rgb_img[:,:,0:3],
                )
        else:
            imsave(
                os.path.join(path_out, 'postpro', new_name)+'.jpg',
                img_out.data,
                )
    elif out_format in [".FIT",".fit",".FITS",".fits"]:
        img_out.write(post_path / fname, overwrite=True)
    else:
        print('Error: Output format not known :-(')


    #   Add image to image list
    img_list.append(rgb_img[:,:,0:3])

sys.stdout.write("\n")

###
#   Write video
#
if mk_video:
    #   Write status to console
    sys.stdout.write("\rWrite video...")
    sys.stdout.write("\n")

    aux.write_video(
        video_name+'.webm',
        img_list,
        video_annotation,
        fps,
        depth=8,
        )

