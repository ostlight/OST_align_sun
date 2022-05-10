#! /usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os 
from os.path import (join, basename)  
import glob
from PIL import Image

import matplotlib.pyplot as plt

from tifffile import (imread, imsave, imshow)

# filepaths
#fp_in = "/path/to/image_*.png"
#fp_in = "output/*.tiff"
fp_in = "output"
fp_out = "se_full.gif"


def mkfilelist(path):
    '''
        Fill the file list

        INPUTS:
            path    - Path to the files - String
    '''
    fileList = os.listdir(path)
    fileList.sort()

    #   Remove not TIFF entries
    tempList = []
    for i in range(0, len(fileList)):
        if(fileList[i].find(".tiff")!=-1 or
            fileList[i].find(".TIFF")!=-1):
                #tempList.append("%s"%(fileList[i]))
                tempList.append(join(path,fileList[i]))
    fileList=tempList
    nfiles=int(len(fileList))

    return fileList, nfiles

fileList, nfiles = mkfilelist(fp_in)
im = imread(fileList, key=0)
print(im.shape)
imshow(im[0])
plt.show()
sys.exit()


# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
#img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

imgs = []
for i in range(0, nfiles):
    if i == 0:
        img = Image.fromarray(im[i])
    else:
        imgs.append(Image.fromarray(im[i]))
        
#imshow(img)
#plt.show()
        
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=50, loop=0, optimize=False)
