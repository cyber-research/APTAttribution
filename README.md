# APT Attribution Code
The code in this repository has been used to benchmarks two Machine Learning approaches performing authorship attribution on a dataset with over 3,500 state-sponsored malware samples.

### Instructions
1. Place the contents of this repository in a folder named `code`.
2. Place the dataset in a folder `dataset` next to the folder `code`.
3. Install all required packages by running `pip3 install -r requirements.txt` and `install_required_packages.sh`.
4. Set the path of your Cuckoo-instance in the files `submit_samples.py` and `retrieve_reports.py` in the folder `cuckoo_api`.
5. Execute `prepare_data.sh` to submit the files to Cuckoo, fetch the reports and preprocess the fetched reports.
6. Execute `run_classifier.sh` to run the experiments. The results can be found in `classifier/results`.

### Dataset Used for Running the Experiment
The dataset that was used for these experiments can be found at GitHub: [APT Malware Dataset](https://github.com/cyber-research/APTMalware "APT Malware Dataset").

### License
**MIT License**

Copyright &copy; 2019 cyber-research

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

