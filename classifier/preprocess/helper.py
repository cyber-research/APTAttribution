import os
from pathlib import Path
import numpy as np
import pandas
import shutil

preprocessor_path = Path(os.path.abspath(__file__)).parent
top_path = preprocessor_path.parent.parent.parent
data_set_path = top_path.joinpath("dataset")
report_path = top_path.joinpath("reports")

temp_path = preprocessor_path.parent.joinpath("tmp")
text_reports_path = temp_path.joinpath("text_reports")
result_path = temp_path.joinpath("preprocessed")


def initalize_paths():

    if not os.path.exists(temp_path):
        os.makedirs(str(temp_path))

    if not os.path.exists(text_reports_path):
        os.makedirs(str(text_reports_path))
    else:
        shutil.rmtree(str(text_reports_path))
        os.makedirs(str(text_reports_path))

    if not os.path.exists(result_path):
        os.makedirs(str(result_path))


def read_dataset_stats():
    with open(str(data_set_path.joinpath('overview.csv'))) as csvfile:
        dataframe = pandas.read_csv(csvfile, index_col='SHA256')
    return dataframe


def print_file_stats(result_files: [(str, Path)], category_counts: {}):
    print('Read and parsed the following files:')
    for key in category_counts.keys():
        print(key.capitalize(), '-', category_counts[key], 'samples -',
              category_counts[key] / len(result_files) * 100, '%')


def json_extract_words(container, bag):
    if isinstance(container, dict):
        for k, v in container.items():
            if isinstance(v, dict):
                bag.append(str(k))
                json_extract_words(v, bag)
            elif isinstance(v, list):
                bag.append(str(k))
                json_extract_words(v, bag)
            else:
                bag.append(str(k))
                bag.append(str(v))
    elif isinstance(container, list):
        for item in container:
            if isinstance(item, dict) or isinstance(item, list):
                json_extract_words(item, bag)
            else:
                bag.append(str(item))
    elif container is not None:
        print(type(container))
        raise ValueError('Something went terribly wrong with flattening the reports! E001')


def csr_vappend(a, b):
    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a
