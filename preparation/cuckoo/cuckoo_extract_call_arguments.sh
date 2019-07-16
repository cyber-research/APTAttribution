#!/bin/bash

for i in ./../reports/*/cuckoo/*; do
	basename=${i##*/}
	newfile=${basename%%.*}.txt
	newdir=${i%/*/*}/cuckoo_extracted_arguments
	mkdir -p "$newdir"
	jq -f ./preparation/cuckoo/cuckoo_extract_call_arguments.jq "$i" > "$newdir/$newfile"
done

