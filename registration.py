
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
from skimage.io import imread_collection, imsave, imshow

import checks
import aux

############################################################################
####                        Routines & definitions                      ####
############################################################################

def check_dirs(path_in, path_out):
    '''
        Check directories

        Parameters
        ----------
        path_in         : `list` of `string`
            Paths to the images

        path_out        : `string`
            Path to the directory where to save the trimmed images

        Returns
        -------
        trim_path       : `pathlib.Path` object
            Path to save the modified images

        path_in_check   : `list` of `pathlib.Path` objects
            Sanitized paths to the input images

    '''
    #   Check directories
    sys.stdout.write("\rCheck directories...\n")
    path_in_check = []
    for path in path_in:
        path_in_check.append(checks.check_pathlib_Path(path))
    checks.check_out(path_out)

    #   Path to the trimmed images
    trim_path = Path(Path(path_out) / 'cut')
    trim_path.mkdir(exist_ok=True)

    return path_in_check, trim_path


def check_bit_depth(data):
    '''
        Check bit depth of the data

        Parameters
        ----------
        data        : `numpy.ndarray`
            Data to investigate

        Returns
        -------
        bit_depth   : `integer`
            Bit depth of the data.
    '''
    if data == 'uint8':
        bit_depth = 8
    elif data == 'uint16':
        bit_depth = 16
    else:
        print('Caution: Bit depth could not be determined.')
        print('Data type is ',data)
        print('Bit depth set to 8')
        bit_depth = 8

    return bit_depth


def calculate_img_mask(bool_mask, nx, ny, upper_right=False, lower_right=False,
                       lower_left=False, mask_points=[[0, 0]]):
    '''
        Calculate image mask

        Parameters
        ----------
        bool_mask   : `boolean`
            If ``True`` a image mask is calculated, otherwise a placeholder
            masked is returned.

        nx          : `integer`
            Image dimension in X direction

        ny          : `integer`
            Image dimension in Y direction

        upper_right     : `boolean`, optional
            Special point for the bool_mask. The upper right corner of the
            image.

        lower_right     : `boolean`, optional
            Special point for the bool_mask. The lower right corner of the
            image.

        lower_left      : `boolean`, optional
            Special point for the bool_mask. The lower left corner of the
            image.

        mask_points     : `list` of `list` of `integer`, optional
            Points to define the bool_mask. The area enclosed by the points
            will be masked.
            Default is ``[[0, 0]]``.

        Returns
        -------
        mask            : `numpy.ndarray`
            Mask area
    '''
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
    else:
        mask = np.ones((ny,nx), dtype=bool)

    return mask


def plot_ref_img_mask(data, mask):
    '''
        Plot the image data and the image mask for comparison

        Parameters
        ----------
        data        : `numpy.ndarray`
            Image data

        mask        : `numpy.ndarray`
            Image mask
    '''
    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

    ax1.imshow(data, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(mask, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Image mask')

    plt.show()


def plot_ref_trim_img(img_ref, img_out):
    '''
        Plot reference and trimmed image

        Parameters
        ----------
        img_ref        : `numpy.ndarray`
            Original image data

        img_out        : `numpy.ndarray`
            Trimmed image mask
    '''
    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

    #ax1.imshow(img_ref.astype('uint8'), cmap='gray')
    ax1.imshow(img_ref, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    #ax2.imshow(img_out.astype('uint8'), cmap='gray')
    ax2.imshow(img_out, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')

    plt.show()


def cal_img_shifts_fits(path_in, path_out, formats, out_format, ref_id=0,
                        mode='cut', bool_mask=False, mask_points=[[0, 0]],
                        upper_right=False, lower_right=False,
                        lower_left=False, bool_heavy=False, offset=0,
                        ys_cut=0, ye_cut=0, xs_cut=0, xe_cut=0,
                        plot_mask=False, plot_cut=False, id_img=0):
    '''
        Calculate image shift for FITS images

        Parameters
        ----------
        path_in         : `list` of `string`
            Paths to the images

        path_out        : `string`
            Path to the directory where to save the trimmed images

        formats         : `list` of `string`
            Allowed input file formats

        out_format      : `string`
            Output format

        ref_id          : `integer`, optional
            ID of the reference image
            Default is ``0``.

        mode            : `string`, optional
            The images can be either cut to the common field of view or extended
            to a field of view in which all images fit
            Possibilities: ``extend`` and ``cut``
            Default is ``cut``.

        bool_mask       : `boolean`, optional
            If ``True`` a image mask is calculated that masks areas, which could
            spoil the cross correlation such as the moon during a solar eclipse.
            Default is `False``.

        mask_points     : `list` of `list` of `integer`, optional
            Points to define the bool_mask. The area enclosed by the points
            will be masked.
            Default is ``[[0, 0]]``.

        upper_right     : `boolean`, optional
            Special point for the bool_mask. The upper right corner of the
            image.

        lower_right     : `boolean`, optional
            Special point for the bool_mask. The lower right corner of the
            image.

        lower_left      : `boolean`, optional
            Special point for the bool_mask. The lower left corner of the
            image.

        bool_heavy      : `boolean`, optional
            If ``True`` a heavyside function will be applied to the image.
            Default is ``False``.

        offset          : `integer`, optional
            Background offset that will be removed from the image.
            Default is ``0``.

        xs_cut          : `integer`, optional
            Number of pixel that are removed from the image from the beginning
            of the image in X direction.
            Default is ``0``.

        xe_cut          : `integer`, optional
            Number of pixel that are removed from the end of the image in
            X direction.
            Default is ``0``.

        ys_cut          : `integer`, optional
            Number of pixel that are removed from the beginning of the image in
            X direction.
            Default is ``0``.

        ye_cut          : `integer`, optional
            Number of pixel that are removed from the end of the image in
            Y direction.
            Default is ``0``.

        plot_mask       : `boolean`, optional
            If ``True`` the image mask and the reference image will be plotted.
            Default is ``False``.

        plot_cut        : `boolean`, optional
            If ``True`` the original reference image in comparison to the
            trimmed reference image will be plotted.
            Default is ``False``.

        id_img          : `integer`, optional
            ID of the reference image to plot by means of plot_cut.
            Default is ``0``.
    '''
    #   Check paths
    path_in_check, trim_path = check_dirs(path_in, path_out)

    #   Create temporary directory
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir=tmp_dir)
    temp_dir_in = tempfile.TemporaryDirectory()

    #   Create links to all files in a temporary directory
    aux.make_symbolic_links(path_in_check, temp_dir_in)

    #   Get file collection
    sys.stdout.write("\rRead images...\n")
    ifc = ccdp.ImageFileCollection(temp_dir_in.name)
    if len(ifc.files) == 0:
        raise RuntimeError('No files found -> EXIT')

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
    bit_depth = check_bit_depth(ref_img_data.dtype)


    ###
    #   Calculate image mask -> the masked area will not be considered,
    #                           while calculating the shifts
    mask = calculate_img_mask(
        bool_mask,
        nx,
        ny,
        upper_right=upper_right,
        lower_right=lower_right,
        lower_left=lower_left,
        mask_points=mask_points,
        )


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
        ref_img_name = ifc_tmp.files[ref_id]
        ref_img_data = CCDData.read(ref_img_name, unit='adu').data
        plot_ref_img_mask(ref_img_data, np.invert(mask))

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

        else:
            print('Error: Mode not known. Check variable "mode".')
            sys.exit()

        #   Set reference image
        if i == ref_id:
            img_ref = img_out.data

        #   Plot reference and offset image
        if i == id_img and plot_cut and ref_id <= id_img:
            plot_ref_trim_img(img_ref, img_out.data)

        i += 1


        ###
        #   Write trimmed image
        #
        default_formats = [
            ".jpg",
            ".jpeg",
            ".JPG",
            ".JPEG",
            ".tiff",
            ".TIFF",
            ]
        new_name = os.path.basename(fname).split('.')[0]
        if out_format in [".FIT",".fit",".FITS",".fits"]:
            img_out.write(trim_path / fname, overwrite=True)
        elif out_format in default_formats:
            imsave(trim_path / str(new_name+out_format), img_out.data)
        else:
            print('Error: Output format not known :-(')

    ##   Remove temporary directory
    #tmp_dir.rmdir()

    sys.stdout.write("\n")


def cal_img_shifts_normal(path_in, path_out, formats, out_format, ref_id=0,
                          mode='cut', bool_mask=False, mask_points=[[0, 0]],
                          upper_right=False, lower_right=False,
                          lower_left=False, bool_heavy=False, offset=0,
                          ys_cut=0, ye_cut=0, xs_cut=0, xe_cut=0,
                          plot_mask=False, plot_cut=False, id_img=0):
    '''
        Calculate image shift for "normal" images, e.g., non FITS images

        Parameters
        ----------
        path_in         : `list` of `string`
            Paths to the images

        path_out        : `string`
            Path to the directory where to save the trimmed images

        formats         : `list` of `string`
            Allowed input file formats

        out_format      : `string`
            Output format

        ref_id          : `integer`, optional
            ID of the reference image
            Default is ``0``.

        mode            : `string`, optional
            The images can be either cut to the common field of view or extended
            to a field of view in which all images fit
            Possibilities: ``extend`` and ``cut``
            Default is ``cut``.

        bool_mask       : `boolean`, optional
            If ``True`` a image mask is calculated that masks areas, which could
            spoil the cross correlation such as the moon during a solar eclipse.
            Default is `False``.

        mask_points     : `list` of `list` of `integer`
            Points to define the bool_mask. The area enclosed by the points
            will be masked.
            Default is ``[[0, 0]]``.

        upper_right     : `boolean`, optional
            Special point for the bool_mask. The upper right corner of the
            image.

        lower_right     : `boolean`, optional
            Special point for the bool_mask. The lower right corner of the
            image.

        lower_left      : `boolean`, optional
            Special point for the bool_mask. The lower left corner of the
            image.

        bool_heavy      : `boolean`, optional
            If ``True`` a heavyside function will be applied to the image.
            Default is ``False``.

        offset          : `integer`, optional
            Background offset that will be removed from the image.
            Default is ``0``.

        xs_cut          : `integer`, optional
            Number of pixel that are removed from the image from the beginning
            of the image in X direction.
            Default is ``0``.

        xe_cut          : `integer`, optional
            Number of pixel that are removed from the end of the image in
            X direction.
            Default is ``0``.

        ys_cut          : `integer`, optional
            Number of pixel that are removed from the beginning of the image in
            X direction.
            Default is ``0``.

        ye_cut          : `integer`, optional
            Number of pixel that are removed from the end of the image in
            Y direction.
            Default is ``0``.

        plot_mask       : `boolean`, optional
            If ``True`` the image mask and the reference image will be plotted.
            Default is ``False``.

        plot_cut        : `boolean`, optional
            If ``True`` the original reference image in comparison to the
            trimmed reference image will be plotted.
            Default is ``False``.

        id_img          : `integer`, optional
            ID of the reference image to plot by means of plot_cut.
            Default is ``0``.
    '''
    #   Check paths
    path_in_check, trim_path = check_dirs(path_in, path_out)

    #   Create links to all files in a temporary directory
    temp_dir_in = tempfile.TemporaryDirectory()
    aux.make_symbolic_links(path_in_check, temp_dir_in)

    #   Make file list
    sys.stdout.write("\rRead images...\n")
    fileList, nfiles = aux.mkfilelist(
        temp_dir_in.name,
        formats=formats,
        addpath=True,
        sort=True,
        )
    if len(fileList) == 0:
        raise RuntimeError('No files found -> EXIT')

    #   Read images
    im = imread_collection(fileList)

    #   Image shape
    image_shape = im[0].shape
    if len(image_shape) > 2:
        color = True
    else:
        color = False
    ny = image_shape[0]
    nx = image_shape[1]
    if color:
        nz = image_shape[2]

    #   Bit depth
    bit_depth = check_bit_depth(im[0].dtype)


    ###
    #   Calculate image mask -> the masked area will not be considered,
    #                           while calculating the shifts
    mask = calculate_img_mask(
        bool_mask,
        nx,
        ny,
        upper_right=upper_right,
        lower_right=lower_right,
        lower_left=lower_left,
        mask_points=mask_points,
        )


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
            if color:
                reff = np.heaviside(
                    im[ref_id][:,:,0]/2**(bit_depth)-offset/2**(bit_depth),
                    0.03,
                    )
                test = np.heaviside(
                    im[i][:,:,0]/2**(bit_depth)-offset/2**(bit_depth),
                    0.03,
                    )
            else:
                reff = np.heaviside(
                    im[ref_id]/2**(bit_depth)-offset/2**(bit_depth),
                    0.03,
                    )
                test = np.heaviside(
                    im[i]/2**(bit_depth)-offset/2**(bit_depth),
                    0.03,
                    )
        else:
            if color:
                reff = im[ref_id][:,:,0]
                test = im[i][:,:,0]
            else:
                reff = im[ref_id]
                test = im[i]

        #   Plot reference image and image mask
        if i == ref_id and plot_mask:
            plot_ref_img_mask(reff, mask)

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
            if color:
                img_i = np.zeros(
                    (ny+deltay, nx+deltax, nz),
                    dtype='uint'+str(bit_depth)
                    )
            else:
                img_i = np.zeros(
                    (ny+deltay, nx+deltax),
                    dtype='uint'+str(bit_depth)
                    )

            #   Set Gamma channel if present
            if color:
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

        #   Plot reference and trimmed image
        if i == id_img and plot_cut and ref_id <= id_img:
            plot_ref_trim_img(img_ref, img_i)

        ##   Write trimmed image
        new_name = os.path.basename(fileList[i]).split('.')[0]
        default_formats = [
            ".jpg",
            ".jpeg",
            ".JPG",
            ".JPEG",
            ".FIT",
            ".fit",
            ".FITS",
            ".fits",
            ]
        if color:
            if out_format in [".tiff", ".TIFF"]:
                imsave(trim_path / str(new_name+out_format), img_i[:,:,0:4])
            elif out_format in default_formats:
                imsave(trim_path / str(new_name+out_format), img_i[:,:,0:3])
            else:
                print('Error: Output format not known :-(')
        else:
            if out_format in default_formats+[".tiff", ".TIFF"]:
                imsave(trim_path / str(new_name+out_format), img_i)
            else:
                print('Error: Output format not known :-(')

    sys.stdout.write("\n")
