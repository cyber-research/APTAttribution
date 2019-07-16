#!/bin/bash

for i in ./../reports/*/cuckoo/*; do
	basename=${i##*/}
	newdir=${i%/*/*}/cuckoo_filtered
	mkdir -p "$newdir"
	jq -f ./preparation/cuckoo/cuckoo_remove_buffer.jq "$i" > "$newdir/$basename"
	sed -i -f ./preparation/cuckoo/cuckoo_remove_max_depth.sed "$newdir/$basename"
done
