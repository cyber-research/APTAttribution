#!/bin/bash

for i in ./../reports/*/cuckoo/*; do
	basename=${i##*/}
	newfile=${basename%%.*}.txt
	newdir=${i%/*/*}/cuckoo_extracted_api
	mkdir -p "$newdir"
	jq -f ./preparation/cuckoo/cuckoo_extract_calls.jq "$i" > "$newdir/$newfile"
done

