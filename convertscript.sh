#!/bin/bash
#Make sure the convert program is installed sudo apt-get install imagemagick
for i in `ls *.jpg`; do convert $i -resize 32x32\! "${i%.jpg}.png"; done
