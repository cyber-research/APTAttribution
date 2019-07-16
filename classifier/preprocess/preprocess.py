import getopt
import sys

from bag_of_words import *
from selector import *
from grouper import *
from helper import *


def preprocess(selector_instance, extractor_instance, group_instance):
    initalize_paths()
    data_set_stats = read_dataset_stats()
    reports_to_read = selector_instance.select()
    print('Start reading files...', end='', flush=True)
    extractor_instance.extract(group_instance, reports_to_read, data_set_stats)
    [(x_train, y_train), (x_test, y_test)] = group_instance.release_results()
    print('done!')
    create_bag_of_words(x_train, y_train, x_test, y_test, type(extractor_instance) is TextExtractor)


def main(argv):
    usage = 'preprocess.py -s <Selector> -g <Groups>'

    try:
        opts, args = getopt.getopt(argv, "s:g:")
    except getopt.GetoptError:
        print(usage)
        exit(2)

    selector_instance = None
    extractor_instance = None
    group_instance = None

    for opt, arg in opts:
        if opt == '-s':
            selector_instance, extractor_instance = selector_switch(arg)
        elif opt == '-g':
            group_instance = group_switch(arg)

    if selector_instance is None:
        print('No valid selector is given!')
        print(usage)
        exit(2)

    if group_instance is None:
        print('No valid group-division is given!')
        print(usage)
        exit(2)

    preprocess(selector_instance, extractor_instance, group_instance)

    with open(str(temp_path.joinpath('signal.txt')), "w") as f:
        f.write('\n'.join([i[0] + ': ' + i[1] for i in opts]))


if __name__ == "__main__":
    main(sys.argv[1:])
