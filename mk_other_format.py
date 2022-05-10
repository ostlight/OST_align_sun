#! /usr/bin/python3
# -*- coding: utf-8 -*-

'''
    Load image of specific formats and convert them to JPEGs
'''

############################################################################
####           Configuration: modify the file in this section           ####
############################################################################

#   Path to the images
path_in  = 'output'

#   Output directory
path_out = 'webp_test'

#   Allowed input file formats
formats = [".tiff", ".TIFF"]

#   Output format
out_format = ".webp"

############################################################################
####                            Libraries                               ####
############################################################################

import sys
import os   

import numpy as np

from tifffile import imread
from skimage.io import imsave

from ost import checks
from ost.reduce import aux    
    
############################################################################
####                               Main                                 ####
############################################################################

#   Check if output directory exists 
checks.check_out(path_out)

#   Make file list
sys.stdout.write("\rRead images...\n")
fileList, nfiles = aux.mkfilelist(path_in, formats=formats, addpath=True)

#   Read images
im = imread(fileList, key=0)


#   Write images to the output directory
for i in range(0,nfiles):
    #   Extract file name
    new_name = os.path.basename(fileList[i]).split('.')[0]
    
    #   Write jpeg
    imsave(fname=os.path.join(path_out,new_name)+out_format, arr=im[i,:,:,0:3])
    
    #   Write status to console
    id_img = i+1
    sys.stdout.write("\rWrite converted image %i" % id_img)
    sys.stdout.flush()
sys.stdout.write("\n")
    
    
