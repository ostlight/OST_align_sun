#!/bin/bash


path='output/webp/*.webp'
frames=''
for f in $path; do
    f=${f//' '/'\ '}
#     echo $f
    frames+="-frame "$f" +100+0+0+0+b "
done
# echo "${frames[@]}" -o animation_test.webp
eval webpmux "${frames[@]}" -o animation_test.webp
