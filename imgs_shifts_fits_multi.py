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
path_in  = [
    '2022-05-05-1441_7-1-CapObj/2022-05-05-1441_7-1-CapObj/',
    '2022-05-05-1450_7-1-CapObj/2022-05-05-1450_7-1-CapObj/',
    '2022-05-05-1456_5-1-CapObj/2022-05-05-1456_5-1-CapObj/',
    '2022-05-05-1506_9-1-CapObj/2022-05-05-1506_9-1-CapObj/',
    '2022-05-05-1515_0-1-CapObj/2022-05-05-1515_0-1-CapObj/'
    ]

#   Output directory
path_out = 'out_multi/'

#   Allowed input file formats
#formats = [".tiff", ".TIFF"]
#formats = [".FIT",".fit",".FITS",".fits"]

#   Output formal
#out_format = ".tiff"
#out_format = ".jpg"
out_format = ".fit"

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
mask_points = [[800,0], [800,2048]]

#   Add a point for the upper right corner
upper_right = False
#   Add a point for the lower right corner
lower_right = True
#   Add a point for the lower left corner
lower_left  = True


###
#   Apply a heavyside function to the image
#
bool_heavy = False
bool_heavy = True

#   Background offset
offset = 11000


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
plot_cut  = False
#   ID of the image to plot
id_img    = 19


############################################################################
####                            Libraries                               ####
############################################################################

import sys
import os

import tempfile

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import ccdproc as ccdp

from astropy.nddata import CCDData

from astropy import log
log.setLevel('ERROR')

import warnings
warnings.filterwarnings('ignore')

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.draw import polygon2mask
from skimage.io import imread, imread_collection, imsave, imshow

import checks
import aux

############################################################################
####                               Main                                 ####
############################################################################

#   Check directories
sys.stdout.write("\rCheck directories...\n")
path_in_check = []
for path in path_in:
    path_in_check.append(checks.check_pathlib_Path(path))
checks.check_out(path_out)

#   Path to the trimmed images
trim_path = Path(Path(path_out) / 'cut')
trim_path.mkdir(exist_ok=True)

#   Create temporary directory
temp_dir_in = tempfile.TemporaryDirectory(dir="tmp")
temp_dir    = tempfile.TemporaryDirectory(dir="tmp")

#   Create links to all files in a temporary directory
aux.make_symbolic_links(path_in_check, temp_dir_in)

#   Get file collection
sys.stdout.write("\rRead images...\n")
ifc = ccdp.ImageFileCollection(temp_dir_in.name)

#   Apply filter to the image collection
#   -> This is necessary so that the path to the image directory is added
#      to the file names. This is required for `calculate_image_shifts`.
ifc = ifc.filter(SIMPLE=True)

#   Number of files
nfiles = len(ifc.files)

#   Image shape
nx = ifc.summary['naxis1'][0]
ny = ifc.summary['naxis2'][0]

#   Get reference image
ref_img_name = ifc.files[ref_id]
ref_img_data = CCDData.read(ref_img_name, unit='adu').data

#   Bit depth
if ref_img_data.dtype == 'uint8':
    bit_depth = 8
elif ref_img_data.dtype == 'uint16':
    bit_depth = 16
else:
    print('Caution: Bit depth could not be determined.')
    print('Data type is ', ref_img_data.dtype)
    print('Bit depth set to 16')
    bit_depth = 16


###
#   Calculate image mask -> the masked area will not be considered,
#                           while calculating the shifts
mask = np.zeros((ny,nx), dtype=bool)
if bool_mask:
    sys.stdout.write("\rCalculate image mask...\n")
    if upper_right:
        mask_points.append([0,nx])
    if lower_right:
        mask_points.append([ny,nx])
    if lower_left:
        mask_points.append([ny,0])

    #   Calculate mask from the specified points
    mask = polygon2mask((ny,nx), mask_points)


###
#   Prepare images and save modified images in the temporary directory
#
for ccd, fname in ifc.ccds(ccd_kwargs=dict(unit='adu'), return_fname=True):
    #   Get mask and add mask from above
    try:
        ccd.mask = ccd.mask | mask
    except:
        ccd.mask = mask

    #   Get image data
    data = ccd.data

    #   "Normalize" image & calculate heavyside function for the image
    if bool_heavy:
        data = np.heaviside(data/2**(bit_depth)-offset/2**(bit_depth), 0.03)

        ccd.data = data

    ccd.write(temp_dir.name+'/'+fname, overwrite=True)


###
#   Calculate shifts
#
sys.stdout.write("\rCalculate shifts...\n")

#   Create new image collection
ifc_tmp = ccdp.ImageFileCollection(temp_dir.name)

#   Apply filter to the image collection
#   -> This is necessary so that the path to the image directory is added
#      to the file names. This is required for `calculate_image_shifts`.
ifc_tmp = ifc_tmp.filter(SIMPLE=True)

#   Plot reference image and image mask
if plot_mask:
    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

    ref_img_name = ifc_tmp.files[ref_id]
    ref_img_data = CCDData.read(ref_img_name, unit='adu').data

    ax1.imshow(ref_img_data, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(np.invert(mask), cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Image mask')

    plt.show()

#   Calculate shifts
shift, flip, minx, maxx, miny, maxy = aux.calculate_image_shifts(
    ifc_tmp,
    ref_id,
    '\tDisplacement for images:',
    method='skimage',
    )


###
#   Cut pictures and add them
#
#   Loop over and trim all images
i = 0
img_list = []
for img_ccd, fname in ifc.ccds(ccd_kwargs={'unit': 'adu'}, return_fname=True):
    #   Write status to console
    sys.stdout.write("\rApply shift to images %i/%i" % (i+1, nfiles))
    sys.stdout.flush()

    #   Trim images, if using phase correlation
    if mode == 'cut':
        #   Flip image if pier side changed
        if flip[i]:
            img_ccd = ccdp.transform_image(img_ccd, np.flip, axis=0)
            img_ccd = ccdp.transform_image(img_ccd, np.flip, axis=1)

        #   Trim images
        verbose=False
        img_out = aux.trim_core(
            img_ccd,
            i,
            nfiles,
            maxx,
            minx,
            maxy,
            miny,
            shift,
            verbose,
            xs_cut=xs_cut,
            xe_cut=xe_cut,
            ys_cut=ys_cut,
            ye_cut=ye_cut,
            )
    elif mode == 'extend':
        print('Sorry the extend mode is not yet implemented. -> EXIT')
        sys.exit()
        ##   Define larger image array
        #img_i = np.zeros(
            #(ny+deltay, nx+deltax, nz),
            #dtype='uint'+str(bit_depth)
            #)

        ##   Set Gamma channel if present
        #if nz == 4:
            #img_i[:,:,3] = 2**bit_depth-1

        ##   Calculate start indexes for the larger image array
        #ys =  maxy - shifts[0,i]
        #xs =  maxx - shifts[1,i]

        ##   Actual image manipulation
        #img_i[ys:ys+ny, xs:xs+nx] = im[i]

    else:
        print('Error: Mode not known. Check variable "mode".')
        sys.exit()

    #   Set reference image
    if i == ref_id:
        img_ref = img_out.data

    #   Plot reference and offset image
    if i == id_img and plot_cut and ref_id <= id_img:
        fig = plt.figure(figsize=(12, 7))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

        ax1.imshow(img_ref, cmap=None)
        ax1.set_axis_off()
        ax1.set_title('Reference image')

        ax2.imshow(img_out.data)
        ax2.set_axis_off()
        ax2.set_title('Offset image')

        plt.show()

    i += 1


    ###
    #   Write trimmed image in FITS format
    #
    img_out.write(trim_path / fname, overwrite=True)


sys.stdout.write("\n")










