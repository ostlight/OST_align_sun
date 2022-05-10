#!/bin/bash


path='output/'
# frames=''
filelist=`ls -b ${path}*'.tiff'`
echo $filelist
for f in $filelist; do
#     echo
    echo "$f"
#     fbase=`echo $f | cut -d "/" -f 2`
#     echo "$fbase"
# #     convert ${f} -define webp:lossless=true ${path}'/webp/'$fbase'.webp'
done
