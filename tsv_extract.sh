#!/bin/bash

# a[$7] in awk means an array with index as string that is in $7. Storing $3 as
# value allows us to print cluster name later by checking 4th field of tsv file
# against index names in a

# create a cluster specific lines from tsv files
# mkdir -p frag_out
# FILES=fragment_files/*
# for f in $FILES
# do
# 	awk -F '\t' 'NR==FNR{a[$7];next} $4 in a' dendritic_barcodes $f  > frag_out/$(basename $f)
# 	echo "$f done!"
# done

# create cluster name files
# FILES=tsv_frag_out/*
# for f in $FILES
# do
#   filename=$(basename $f)
#   filename=${filename#*_} #removes everything before first occurence of _
#   filename=${filename%_*} #removes everything after last occurence of _
# 	awk -F '\t' 'NR==FNR{a[$7]=$3;next} $4 in a {print "'$filename'" "#" $4 "," a[$4]}' dendritic_barcodes $f > cluster_names/$filename
#   sed -i '1s/^/cell,cluster\n/' cluster_names/$filename
#   echo "$f done!"
# done

#create bed files for SCATE
# FILES=tsv_frag_out/*
# input="dendritic_barcodes"
# cells=$(awk -F '\t' '{print $7}' $input)
# for cell in $cells
# do
#   for f in $FILES
#   do
#     filename=$(basename $f)
#     filename=${filename#*_} #removes everything before first occurence of _
#     filename=${filename%_*} #removes everything after last occurence of _
#   	awk -F '\t' '$4 == "'$cell'" {print $1, $2, $3, NR "#" "'$filename'" }'  $f >> scate_bed_files/"$cell.bed"
#   done
# done



#complete flow for scate bam
FILES=fragment_files/*
folder="scate_files/progen" #"scate_files/B_cells" #"mono"
mkdir -p $folder/bed_files
# create bed files out of fragment files where cell barcode matches the barcodes of interest
for f in $FILES
do
  filename=$(basename $f)
  filename=${filename#*_} #removes everything before first occurence of _
  filename=${filename%_*} #removes everything after last occurence of _
	awk -F '\t' 'NR==FNR{a[$7];next} $4 in a {print $1"\t"$2"\t"$3"\t"NR "#" "'$filename'" }'  $folder/barcodes $f  > $folder/bed_files/"$filename.bed"
	echo "$f done!"
done
#bed to bam conversion
FILES=${folder}/bed_files/*
mkdir -p $folder/bam_files
for f in $FILES
do
  filename=$(basename $f)
  filename=${filename%.*}
  tr ' ' '\t' < $f 1<> $f # to convert single space to tab
  bedToBam -i $f -g /home/eramamur/resources/genomes/hg19/hg19.chrom.sizes > $folder/bam_files/"$filename.bam"
done
#to remove all bed files with 0 lines
find ${folder}/bed_files/ -type f -exec awk -v x=1 'NR==x{exit 1}' {} \; -exec rm -f {} \;
# to remove corresponding bam FILES
find ${folder}/bam_files/ -type 'f' -size -126c -delete
