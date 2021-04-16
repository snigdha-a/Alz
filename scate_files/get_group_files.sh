#!/bin/bash

names=$(ls ../fragment_files/)

# groups in barcodes file
awk '{print $4}' ../GSE129785_scATAC-Hematopoiesis-All.cell_barcodes.txt | sort | uniq > groups
# all group name based files in the raw tar
for f in $names ; do d=${f#*_} ; d=${d%_*} ;echo $d ; done | sort > filenames
# to get files for matching groups. Ignoring the groups for which no file exist.
matching_groups=$(comm -12 filenames groups)
# getting the whole filenames of every matching group
filtered_files=$(for a in $matching_groups ; do ls ../fragment_files/ | grep $a; done)

#monocyte barcodes
grep -E 'Cluster12|Cluster13' ../GSE129785_scATAC-Hematopoiesis-All.cell_barcodes.txt | awk '{print $7}'> mono/monocyte_barcodes
# look for each barcode in monocyte in each of the filtered files
while read -r line; do
  for a in $filtered_files
  do
    LC_ALL=C grep $line ../fragment_files/$a >> proper/mono/"$line.bed"
  done
  echo $line "done"
done < mono/monocyte_barcodes
