
############################################################################
####                            Libraries                               ####
############################################################################

import numpy as np

from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

import aux

############################################################################
####                        Routines & definitions                      ####
############################################################################


def postprocess(img_data, bit_depth, global_histo_equalization,
                local_histo_equalization, disk_size,
                adaptive_histo_equalization, clip_limit, log_cont_adjust,
                log_gain, gamma_adjust, gamma, contrast_stretching,
                upper_percentile, lower_percentile, postprocess_layers):
    '''
        Postprocess the image

        Parameters
        ----------
        img_data                    : `numpy.ndarray`
            Image data

        bit_depth                   : `integer`
            Bit depth of the image

        global_histo_equalization   : `boolean`
            If ``True`` a global histogram equalization will be applied
            to the image.

        local_histo_equalization    : `boolean`
            If ``True`` a local histogram equalization will be applied
            to the image.

        disk_size                   : `integer`
            Footprint for local histogram equalization (disk morphology is used)

        adaptive_histo_equalization : `boolean`
            If ``True`` a adaptive histogram equalization will be applied
            to the image.

        clip_limit                  : `float`
            Clip limit for adaptive histogram equalization

        log_cont_adjust             : `boolean`
            If ``True`` log contrast adjustment will be applied to the image.

        log_gain                    : `float`
            Gain for the log contrast adjustment.

        gamma_adjust                : `boolean`
            If ``True`` a gamma contrast adjustment will be applied to the
            image.

        gamma                       : `float`
            Gamma value for the contrast adjustment.

        contrast_stretching         : `boolean`
            If ``True`` a contrast stretching will be applied to the image.

        upper_percentile            : `float`
            Upper percentile for contrast stretching

        lower_percentile            : `float`
            Lower percentile for contrast stretching

        postprocess_layers          : `list`
            List with parameters for the postprocessing by means of the planetary system stacker

        Returns
        -------
        img_data                    : `numpy.ndarray`
            Postprocess image data
    '''
    ###
    #   Postprocessing
    #

    #   Global histogram equalize
    if global_histo_equalization:
        img_glob_eq = exposure.equalize_hist(img_data)
        img_data = img_glob_eq*2**(bit_depth)

    #   Local equalization
    if local_histo_equalization:
        footprint = disk(disk_size)
        img_eq = rank.equalize(img_data, footprint)
        img_data = img_eq

    #   Adaptive histogram equalization
    if adaptive_histo_equalization:
        img_adapteq = exposure.equalize_adapthist(
            img_data,
            clip_limit=clip_limit,
            )
        img_data = img_adapteq*2**(bit_depth)

    #   log contrast adjustment
    if log_cont_adjust:
        logarithmic_corrected = exposure.adjust_log(img_data, log_gain)
        img_data = logarithmic_corrected

    #   Gamma contrast adjustment
    if gamma_adjust:
        gamma_corrected = exposure.adjust_gamma(img_data, gamma)
        img_data = gamma_corrected

    #   Contrast stretching
    if contrast_stretching:
        plow, pup = np.percentile(
            img_data,
            (lower_percentile, upper_percentile),
            )
        img_rescale = exposure.rescale_intensity(
            img_data,
            in_range=(plow, pup),
            )
        img_data = img_rescale


    ###
    #   Sharp the image
    #
    #   Default layer
    layers = [aux.PostprocLayer(1., 1., 0., 20, 0., False)]

    #   Add user layer
    for layer in postprocess_layers:
        layers.append(aux.PostprocLayer(*layer))

    #   Sharp/prostprocess image
    img_data = aux.post_process(img_data, layers)

    return img_data

