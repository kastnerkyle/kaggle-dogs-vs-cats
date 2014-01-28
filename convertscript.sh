#!/bin/bash
#Make sure the convert program is installed sudo apt-get install imagemagick
for i in `ls *.jpg`; do convert $i -resize 221x221\! "${i%.jpg}.png"; done
