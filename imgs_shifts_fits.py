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
path_in  = '2022-05-05-1439_9-1-CapObj/2022-05-05-1439_9-1-CapObj/'
#path_in  = '2022-05-05-1439_9-1-CapObj/test/'
#path_in  = '2022-05-05-1439_9-1-CapObj/test_2/'

#   Output directory
path_out = 'out_Halpha/'

#   Allowed input file formats
#formats = [".tiff", ".TIFF"]
formats = [".FIT",".fit",".FITS",".fits"]

#   Output formal
out_format = ".tiff"
out_format = ".jpg"
#out_format = ".fit"

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
##   Upper edge
#ys_cut = 350
##   Lower edge
#ye_cut = 320
##   Left edge
#xs_cut = 580
##   Right edge
#xe_cut = 570

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

#   Local histogram equalization
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


###
#   Video options
#
#   Make a video (True or False)?
mk_video = True
#   Video name
video_name = 'test_Halpha'
#   Video annotation
video_annotation = ''
#   Frames per second
fps = 20


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
#from astropy import log
#log.setLevel('ERROR')

import warnings
warnings.filterwarnings('ignore')

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.draw import polygon2mask
from skimage.io import imread, imread_collection, imsave, imshow

from skimage.util import img_as_ubyte
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
out_path = Path(Path(path_out) / 'cut')
out_path.mkdir(exist_ok=True)

#   Create temporary directory
temp_dir = tempfile.TemporaryDirectory()

#   Get file collection
sys.stdout.write("\rRead images...\n")
ifc = ccdp.ImageFileCollection(path_in)

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
    sys.stdout.write("\rApply shift to image %i/%i" % (i+1, nfiles))
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
            len(ifc.files),
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
    #   Postprocessing
    #

    #print('---------------------')
    #print(img_out.data)
    #img_out.data = aux.post_process(img_out.data, layers)
    #print(processed_data)
    #print('=============================')

    #print(img_out.data)
    #print('-----------------')

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
    #   Write image
    #
    new_name = 'shift_'+os.path.basename(fname).split('.')[0]
    if out_format in [".tiff", ".TIFF"]:
        if mkRGB:
            imsave(
                os.path.join(path_out, 'cut', new_name)+'.tiff',
                rgb_img,
                )
        else:
            imsave(
                os.path.join(path_out, 'cut', new_name)+'.tiff',
                img_out.data,
                )
    elif out_format in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        if mkRGB:
            imsave(
                os.path.join(path_out, 'cut', new_name)+'.jpg',
                rgb_img[:,:,0:3],
                )
        else:
            imsave(
                os.path.join(path_out, 'cut', new_name)+'.jpg',
                img_out.data,
                )
    elif out_format in [".FIT",".fit",".FITS",".fits"]:
        img_out.write(out_path / fname, overwrite=True)
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
        video_name+'.mp4',
        #'test_Halpha.mpeg',
        #'test_Halpha.mkv',
        #'test_Halpha.webm',
        #'test_Halpha.avi',
        #'test_Halpha.divx',
        #'test_Halpha.vp09',
        img_list,
        video_annotation,
        fps,
        depth=8,
        )









