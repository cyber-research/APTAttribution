#!/bin/bash

echo "Submitting samples to Cuckoo..."
python3 cuckoo_api/submit_samples.py
echo
read -p "Press Return when Cuckoo has finished processing all samples..."
mkdir -p ./../reports;
echo
echo "Downloading reports from Cuckoo..."
python3 cuckoo_api/retrieve_reports.py
echo
echo "Preparing Cuckoo reports..."
source ./preparation/cuckoo/cuckoo_filter.sh
source ./preparation/cuckoo/cuckoo_extract_calls.sh
source ./preparation/cuckoo/cuckoo_extract_call_arguments.sh
echo
echo "Finished!"
