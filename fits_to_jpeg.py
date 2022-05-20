#! /usr/bin/python3
# -*- coding: utf-8 -*-

'''
    Convert fits to jpeg
'''


############################################################################
####           Configuration: modify the file in this section           ####
############################################################################

#   Path to the images
path_in  = '2022-05-05-1441_7-1-CapObj/2022-05-05-1441_7-1-CapObj/'

#   Output directory
path_out = '2022-05-05-1441_7-1-CapObj/'

############################################################################
####                            Libraries                               ####
############################################################################

import sys
import os

from pathlib import Path

import ccdproc as ccdp

from skimage.io import imsave

import checks
import aux

############################################################################
####                               Main                                 ####
############################################################################

#   Check directories
sys.stdout.write("\rCheck directories...\n")
path_in  = checks.check_pathlib_Path(path_in)
checks.check_out(path_out)
out_path = Path(Path(path_out) / 'jpeg')
out_path.mkdir(exist_ok=True)

#   Get file collection
sys.stdout.write("\rRead images...\n")
ifc = ccdp.ImageFileCollection(path_in)

#   Apply filter to the image collection
#   -> This is necessary so that the path to the image directory is added
#      to the file names. This is required for `calculate_image_shifts`.
ifc = ifc.filter(SIMPLE=True)

#   Loop over all images
for ccd, fname in ifc.ccds(ccd_kwargs=dict(unit='adu'), return_fname=True):
    new_name = os.path.basename(fname).split('.')[0]
    imsave(
        os.path.join(path_out, 'jpeg', new_name)+'.jpg',
        ccd.data,
        )
