
############################################################################
####                            Libraries                               ####
############################################################################

import os

import numpy as np

import ccdproc as ccdp

from astropy.nddata import CCDData

from skimage.registration import phase_cross_correlation


############################################################################
####                        Routines & definitions                      ####
############################################################################


def mkfilelist(path, formats=[".FIT",".fit",".FITS",".fits"], addpath=False,
               sort=False):
    '''
        Fill the file list

        Parameters
        ----------
        path            : `string'
            Path to the files

        formats         : `list' of `strings', optional
            Allowed Formats
            Default is ``[".FIT",".fit",".FITS",".fits"]``.

        addpath         : `boolean`, optional
            If True the path will be added to  the file names.
            Default is ``False``.

        sort            : `boolean`, optional
            If True the files will be sorted.
            Default is ``False``.

        Returns
        -------
        fileList        : `list' of `strings'
            List with file names

        nfiles          : `integer`
            Number of files.
    '''
    fileList = os.listdir(path)
    if sort:
        fileList.sort()

    #   Remove not TIFF entries
    tempList = []
    for i in range(0, len(fileList)):
        for j, form in enumerate(formats):
            if fileList[i].find(form)!=-1:
                    if addpath:
                        tempList.append(os.path.join(path,fileList[i]))
                    else:
                        #tempList.append("%s"%(fileList[i]))
                        tempList.append(fileList[i])
    fileList=tempList
    nfiles=int(len(fileList))

    return fileList, nfiles



def cal_delshifts(shifts, pythonFORMAT=False):
    '''
        Calculate shifts

        Parameters
        ----------
        shifts              : `numpy.ndarray`
            2D numpy array with the image shifts in X and Y direction

        pythonFORMAT        : `boolean`
            If True the python style of image ordering is used. If False the
            natural/fortran style of image ordering is use.
            Default is ``False``.

        Returns
        -------
        minx            : `float'
            Minimum shift in X direction

        maxx            : `float'
            Maximum shift in X direction

        miny            : `float'
            Minimum shift in Y direction

        maxy            : `float'
            Maximum shift in Y direction

        deltax          : `float'
            Difference between maxx and minx

        deltay          : `float'
            Difference between maxy and miny
    '''
    #   Distinguish between python format and natural format
    if pythonFORMAT:
        id_x = 1
        id_y = 0
    else:
        id_x = 0
        id_y = 1

    #   Maximum and minimum shifts
    minx = min(shifts[id_x,:])
    maxx = max(shifts[id_x,:])

    miny = min(shifts[id_y,:])
    maxy = max(shifts[id_y,:])

    #   Delta between the shifts
    deltax = maxx - minx
    deltay = maxy - miny

    return minx, maxx, miny, maxy, deltax, deltay


def cal_delshifts_new(shifts, pythonFORMAT=False):
    '''
        Calculate shifts

        Parameters
        ----------
        shifts              : `numpy.ndarray`
            2D numpy array with the image shifts in X and Y direction

        pythonFORMAT        : `boolean`
            If True the python style of image ordering is used. If False the
            natural/fortran style of image ordering is use.
            Default is ``False``.

        Returns
        -------
        minx            : `float'
            Minimum shift in X direction

        maxx            : `float'
            Maximum shift in X direction

        miny            : `float'
            Minimum shift in Y direction

        maxy            : `float'
            Maximum shift in Y direction


    '''
    #   Distinguish between python format and natural format
    if pythonFORMAT:
        id_x = 1
        id_y = 0
    else:
        id_x = 0
        id_y = 1

    #   Maximum and minimum shifts
    minx = min(shifts[id_x,:])
    maxx = max(shifts[id_x,:])

    miny = min(shifts[id_y,:])
    maxy = max(shifts[id_y,:])

    return minx, maxx, miny, maxy


def shiftsTOindex(shifts, minx, miny, nxmax, nymax, deltax, deltay,
                  pythonFORMAT=False):
    '''
        Translate image shifts to image index positions

        Parameters
        ----------
        shift           : `numpy.ndarray`
            Shift of this specific image in X and Y direction

        minx            : `float'
            Minimum shift in X direction

        miny            : `float'
            Minimum shift in Y direction

        nxmax           : `float'
            Maximum shift in X direction

        nymax           : `float'
            Maximum shift in Y direction

        deltax          : `float'
            Difference between maxx and minx

        deltay          : `float'
            Difference between maxy and miny

        pythonFORMAT    : `boolean', optional
            If True python format for array index will be use, otherwise FITS
            natural format is used.
            Default is ``False``.

        Returns
        -------
        xs              : `integer`
            Start index to cut the image in X direction

        xe              : `integer`
            End index to cut the image in X direction

        ys              : `integer`
            Start index to cut the image in Y direction

        ye              : `integer`
            End index to cut the image in Y direction

    '''
    #   Distinguish between python format and natural format
    if pythonFORMAT:
        id_x = 1
        id_y = 0
    else:
        id_x = 0
        id_y = 1

    #   Calculate shifts
    xs = 0 + shifts[id_x] - minx
    xe = nxmax - (deltax-(shifts[id_x]-minx))
    ys = 0 + shifts[id_y] - miny
    ye = nymax - (deltay-(shifts[id_y]-miny))

    return xs, xe, ys, ye


def make_index_from_shifts(nx, ny, maxx, minx, maxy, miny, shift):
    '''
        Calculate image index positions from image shifts

        Parameters
        ----------
        nx              : `float'
            Image dimension in X direction

        ny              : `float'
            Image dimension in Y direction

        maxx            : `float'
            Maximum shift in X direction

        minx            : `float'
            Minimum shift in X direction

        maxy            : `float'
            Maximum shift in Y direction

        miny            : `float'
            Minimum shift in Y direction

        shift           : `numpy.ndarray`
            Shift of this specific image in X and Y direction

        Returns
        -------
        xs, xe, ys, ye  : `float`
            Start/End pixel index in X direction `xs`/`xe` and start/end pixel
            index in Y direction.
    '''
    #   Calculate indexes from image shifts
    xs = maxx - shift[1]
    xe = nx + minx - shift[1]
    ys = maxy - shift[0]
    ye = ny + miny - shift[0]

    #deltax = maxx - minx
    #deltay = maxy - miny
    #xs = 0 + shift[1] - minx
    #xe = nx - (deltax-(shift[1]-minx))
    #ys = 0 + shift[0] - miny
    #ye = ny - (deltay-(shift[0]-miny))

    return xs, xe, ys, ye


def trim_core(img, i, nfiles, maxx, minx, maxy, miny, shift, verbose):
    '''
        Trim image 'i' based on a shift compared to a reference image

        Parameters
        ----------
        img             : `astropy.nddata.CCDData`
            Image

        i               : `integer`
            Number of the image in the sequence

        nfiles          : `integer`
            Number of all images

        nx              : `float'
            Image dimension in X direction

        ny              : `float'
            Image dimension in Y direction

        maxx            : `float'
            Maximum shift in X direction

        minx            : `float'
            Minimum shift in X direction

        maxy            : `float'
            Maximum shift in Y direction

        miny            : `float'
            Minimum shift in Y direction

        shift           : `numpy.ndarray`
            Shift of this specific image in X and Y direction

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.
    '''
    if verbose:
        #   Write status to console
        sys.stdout.write("\r\tApply shift to image %i/%i\n" % (i+1, nfiles))
        sys.stdout.flush()

    #   Calculate indexes from image shifts
    xs, xe, ys, ye = make_index_from_shifts(
        img.shape[1],
        img.shape[0],
        maxx,
        minx,
        maxy,
        miny,
        shift[:,i],
        )

    #   Trim the image
    img = ccdp.trim_image(img[ys:ye, xs:xe])

    return img


def correl_images(img_a, img_b, xMax, yMax, debug):
    '''
        Cross correlation:

        Adapted from add_images written by Nadine Giese for use within the
        astrophysics lab course at Potsdam University.
        The source code may be modified, reused, and distributed as long as
        it retains a reference to the original author(s).

        Idea and further information:
        http://en.wikipedia.org/wiki/Phase_correlation

        Parameters
        ----------
        img_a       : `numpy.ndarray`
            Data of first image

        img_b       : `numpy.ndarray`
            Data of second image

        xMax        : `integer`
            Maximal allowed shift between the images in Pixel - X axis

        yMax        : `integer`
            Maximal allowed shift between the images in Pixel - Y axis

        debug       : `boolean`
            If True additional plots will be created
    '''

    lx = img_a.shape[1]
    ly = img_a.shape[0]


    imafft = np.fft.fft2(img_a)
    imbfft = np.fft.fft2(img_b)
    imbfftcc = np.conj(imbfft)
    fftcc = imafft*imbfftcc
    fftcc = fftcc/np.absolute(fftcc)
    #cc = np.fft.ifft2(fftcc)
    cc = np.fft.fft2(fftcc)
    cc[0,0] = 0.

    for i in range(xMax,lx-xMax):
        for j in range(0,ly):
            cc[j,i] = 0
    for i in range(0,lx):
        for j in range(yMax,ly-yMax):
            cc[j,i] = 0

    #   Debug plot showing the cc matrix
    if debug:
        plot.debug_plot_cc_matrix(img_b, cc)
        #from matplotlib import pyplot as plt
        #from astropy.visualization import simple_norm
        #norm = simple_norm(img_b, 'log', percent=99.)
        #plt.subplot(121),plt.imshow(img_b, norm=norm, cmap = 'gray')
        #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        #norm = simple_norm(np.absolute(cc), 'log', percent=99.)
        #plt.subplot(122),plt.imshow(np.absolute(cc), norm=norm,
                                    #cmap = 'gray')
        #plt.title('cc'), plt.xticks([]), plt.yticks([])
        #plt.show()

    #   Find the maximum in cc to identify the shift
    ind1, ind2 = np.unravel_index(cc.argmax(), cc.shape)

    if ind2 > lx/2.:
        ind2 = (ind2-1)-lx+1
    else:
        ind2 = ind2 - 1
    if ind1 > ly/2.:
        ind1 = (ind1-1)-ly+1
    else:
        ind1 = ind1 - 1

    return ind1, ind2


def calculate_image_shifts(ifc, ref_img, comment, method='skimage'):
    '''
        Calculate image shifts

        Parameters
        ----------
        ifc        : `ccdproc.ImageFileCollection`
            Image file collection

        ref_img         : `integer`
            Number of the reference image

        comment         : `string`
            Information regarding for which images the shifts will be
            calculated

        method          : `string`, optional
            Method to use for image alignment.
            Possibilities: 'own'     = own correlation routine based on phase
                                       correlation, applying fft to the images
                           'skimage' = phase correlation with skimage'


        Returns
        -------
        shift           : `numpy.ndarray`
            Shifts of the images in X and Y direction

        flip            : `numpy.ndarray`
            Flip necessary to account for pier flips

        maxx            : `float'
            Maximum shift in X direction

        minx            : `float'
            Minimum shift in X direction

        maxy            : `float'
            Maximum shift in Y direction

        miny            : `float'
            Minimum shift in Y direction

    '''
    #   Number of images
    nfiles = len(ifc.files)

    #   Get reference image, reference mask, and corresponding file name
    reff_name = ifc.files[ref_img]
    try:
        reff_ccd  = CCDData.read(reff_name)
    except:
        reff_ccd  = CCDData.read(reff_name, unit='adu')
    reff_data = reff_ccd.data
    try:
        reff_mask = np.invert(reff_ccd.mask)
    except:
        #reff_mask = True
        reff_mask = np.ones(reff_data.shape, dtype=bool)

    reff_pier = reff_ccd.meta.get('PIERSIDE', 'EAST')

    #   Prepare an array for the shifts
    shift = np.zeros((2,nfiles), dtype=int)
    flip  = np.zeros(nfiles, dtype=bool)

    print(comment)
    print('\tImage\tx\ty\tFilename')
    print('\t----------------------------------------')
    print('\t{}\t{}\t{}\t{}'.format(ref_img, 0, 0, reff_name.split('/')[-1]))

    #   Calculate image shifts
    i = 0
    for ccd, fname in ifc.ccds(ccd_kwargs=dict(unit='adu'), return_fname=True):
        if i != ref_img:
            #   Image and mask to compare with
            test_data = ccd.data
            try:
                test_mask = np.invert(ccd.mask)
            except:
                test_mask = np.ones(test_data.shape, dtype=bool)

            #   Image pier side
            test_pier = ccd.meta.get('PIERSIDE', 'EAST')

            #   Flip if pier side changed
            if test_pier != reff_pier:
                #test_data = transform_image(test_data, np.flip,
                #test_mask = transform_image(test_mask
                test_data = np.flip(test_data, axis=0)
                test_data = np.flip(test_data, axis=1)
                test_mask = np.flip(test_mask, axis=0)
                test_mask = np.flip(test_mask, axis=1)
                flip[i] = True

            #   Calculate shifts
            if method == 'skimage':
                shift[:,i] = phase_cross_correlation(
                    reff_data,
                    test_data,
                    reference_mask=reff_mask,
                    moving_mask=test_mask,
                    )
            elif method == 'own':
                shift[0,i], shift[1,i] = correl_images(
                    reff_data,
                    test_data,
                    1000,
                    1000,
                    #True,
                    False,
                    )
                shift[1,i] = shift[1,i] * -1
                shift[0,i] = shift[0,i] * -1
            else:
                #   This should not happen...
                raise RuntimeError(
                    'Image correlation method '+str(method)+' not known\n'
                    )

            print('\t{}\t{}\t{}\t{}'.format(
                i,
                shift[1,i],
                shift[0,i],
                fname,
                ))
        i += 1
    print()

    #   Calculate maximum, minimum, and delta shifts
    minx, maxx, miny, maxy = cal_delshifts_new(shift, pythonFORMAT=True)

    return shift, flip, minx, maxx, miny, maxy
