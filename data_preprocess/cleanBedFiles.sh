#!/bin/bash
for f in ./bedfiles/*;
do
  sed -i 's/-/  /g' $f
  sed -i 's/\..*//g' $f
done
