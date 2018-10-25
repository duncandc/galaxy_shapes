#!/bin/bash

url="http://sdss.physics.nyu.edu/vagc-dr7/vagc2/sdss/parameters/"
input="./file_list.txt"
while IFS= read -r var
do
  wget "$url$var"
done < "$input"