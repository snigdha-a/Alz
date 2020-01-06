#!/bin/bash
input="out"
outputFile="labels"
while IFS= read -r line
do
	labels=()
	for f in ./bedfiles/*;
	do
		if grep -q $line $f; then
			labels+=1
		else
			labels+=0
		fi 
	done
		echo $labels >> outputFile
done < "$input"
