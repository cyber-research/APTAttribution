import getopt
import sys
from pathlib import Path
from algorithm import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from tensorflow import keras
from sklearn import metrics
import os
import pandas
import time
import gc
import shutil
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

learner_path = Path(os.path.abspath(__file__)).parent
out_path = learner_path.parent.joinpath("output")
processed_path = learner_path.parent.joinpath("tmp", "preprocessed")
splits = 5


def learn(algorithm_instance: Algorithm, undersampling: bool, oversampling: bool, timestamp: str):

    scaler = MinMaxScaler()
    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    run_sets = []

    seperated = Path.exists(processed_path.joinpath('train_labels.txt'))

    # Split both test and training and keep seperated
    if seperated:
        bag_of_words_train = sparse.load_npz(processed_path.joinpath("bag_of_words_train.npz"))
        bag_of_words_test = sparse.load_npz(processed_path.joinpath("bag_of_words_test.npz"))
        y_train_raw = list(map(int, open(str(processed_path.joinpath('train_labels.txt'))).read().splitlines()))
        y_test_raw = list(map(int, open(str(processed_path.joinpath('test_labels.txt'))).read().splitlines()))
        num_classes = np.max([np.max(y_train_raw), np.max(y_test_raw)]) + 1
        x_train_raw = scaler.fit_transform(bag_of_words_train.toarray().astype(float))
        x_test_raw = scaler.fit_transform(bag_of_words_test.toarray().astype(float))

        train_runs = []
        test_runs = []
        runs = []

        for train_index, _ in skf.split(x_train_raw, y_train_raw):
            train_runs.append(train_index)
        for test_index, _ in skf.split(x_test_raw, y_test_raw):
            test_runs.append(test_index)

        for i in range(splits):
            runs.append((train_runs[i], test_runs[i]))

        del bag_of_words_train, bag_of_words_test, train_runs, test_runs

    else:
        bag_of_words = sparse.load_npz(processed_path.joinpath("bag_of_words.npz"))
        y_raw = list(map(int, open(str(processed_path.joinpath('labels.txt'))).read().splitlines()))
        num_classes = np.max(y_raw) + 1
        x_raw = scaler.fit_transform(bag_of_words.toarray().astype(float))
        runs = []
        for train_index, test_index in skf.split(x_raw, y_raw):
            runs.append((train_index, test_index))
        del bag_of_words

    del scaler, skf

    accuracies = []

    for i in range(splits):

        if seperated:  # Seperated train and test set
            (x_train, y_train), (x_test, y_test) = (
                (x_train_raw[runs[i][0]], keras.utils.to_categorical([y_train_raw[j] for j in runs[i][0]], num_classes))
                ,
                (x_test_raw[runs[i][1]], keras.utils.to_categorical([y_test_raw[j] for j in runs[i][1]], num_classes))
            )
        else:
            (x_train, y_train), (x_test, y_test) =\
                (
                    (x_raw[runs[i][0]], keras.utils.to_categorical([y_raw[j] for j in runs[i][0]], num_classes))
                    ,
                    (x_raw[runs[i][1]], keras.utils.to_categorical([y_raw[j] for j in runs[i][1]], num_classes))
                )

        if undersampling:
            rus = RandomUnderSampler()
            x_train, y_train = rus.fit_resample(x_train, y_train)

        if oversampling:
            ros = RandomOverSampler()
            x_train, y_train = ros.fit_resample(x_train, y_train)

        # Compensate for strange behaviour binary case
        if y_train.shape[1] != num_classes:
            y_train = keras.utils.to_categorical(y_train, num_classes)

        algorithm_instance.fit(x_train, y_train)
        y_pred_proba = algorithm_instance.predict_proba(x_test)

        y_pred = [np.argmax(y) for y in y_pred_proba]
        y_pred = keras.utils.to_categorical(y_pred, num_classes)

        col_test = []
        col_pred_proba = []
        col_pred = []

        for j in range(num_classes):
            col_test.append('y_test_%s' % j)
            col_pred_proba.append('y_pred_proba_%s' % j)
            col_pred.append('y_pred_%s' % j)

        columns = col_test
        columns.extend(col_pred_proba)
        columns.extend(col_pred)

        data = np.concatenate((y_test, y_pred_proba), axis=1)
        data = np.concatenate((data, y_pred), axis=1)

        df = pandas.DataFrame(data=data, columns=columns)
        df.to_csv(out_path.joinpath(timestamp, 'run%s.csv' % i))

        accuracies.append(metrics.accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
        del x_train, x_test, y_train, y_test, y_pred_proba, y_pred, data, df
        gc.collect()

    print('Accuracy:', np.average(accuracies))


def main(argv):
    usage = 'learner.py -a <Algorithm> -u <Uniform>'

    try:
        opts, args = getopt.getopt(argv, "a:u:")
    except getopt.GetoptError:
        print(usage)
        exit(2)

    algorithm_instance = None
    undersampling = False
    oversampling = False

    for opt, arg in opts:
        if opt == '-a':
            algorithm_instance = algorithm_switch(arg)
        elif opt == '-u':
            if arg == 'Undersampling':
                undersampling = True
            elif arg == 'Oversampling':
                oversampling = True
            else:
                print('No valid way to make sets uniform is given!')
                print(usage)
                exit(2)

    if algorithm_instance is None:
        print('No valid algorithm is given!')
        print(usage)
        exit(2)

    timestamp = str(int(time.time()))
    os.mkdir(str(out_path.joinpath(timestamp)))

    with open(str(processed_path.parent.joinpath('signal.txt')), 'r') as f:
        config = f.readlines()

    config.append("\n")
    config.append("\n".join([i[0] + ': ' + i[1] for i in opts]))

    with open(str(out_path.joinpath(timestamp, 'config.txt')), "w") as f:
        f.writelines(config)

    shutil.copy(str(processed_path.parent.joinpath('group_labels.txt')),
                str(out_path.joinpath(timestamp, 'group_labels.txt')))

    learn(algorithm_instance, undersampling, oversampling, timestamp)


if __name__ == "__main__":
    main(sys.argv[1:])
