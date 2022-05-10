
############################################################################
####                            Libraries                               ####
############################################################################

import os


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
