from pathlib import Path
import pandas
import numpy as np
from sklearn import metrics
import os
import json
import seaborn as sn
import matplotlib.pyplot as plt

CURRENT_PATH = Path(os.path.abspath(__file__)).parent
OUT_PATH = str(CURRENT_PATH.parent.joinpath('output'))

RECALL = False
FALLOUT = False
PRECISION = False
F1 = False
AUC = False
ACCURACY = True


def calc_metrics(path: Path, result_dict: dict):

    with open(str(path.joinpath('config.txt')), 'r') as f:
        config_lines = f.readlines()
	
    grouper = config_lines[1].replace('-g: ', '').replace('\n', '')
    algorithm = config_lines[2].replace('-a: ', '').replace('\n', '')
    selector = config_lines[0].replace('-s: ', '').replace('\n', '')
    sampling = 'Unbalanced' if len(config_lines) < 4 else config_lines[3].replace(
        '-u: ', '').replace('\n', '')

    if grouper not in result_dict:
        result_dict[grouper] = dict()
	
    if algorithm not in result_dict[grouper]:
        result_dict[grouper][algorithm] = dict()

    if selector not in result_dict[grouper][algorithm]:
        result_dict[grouper][algorithm][selector] = dict()

    if sampling not in result_dict[grouper][algorithm][selector]:
        result_dict[grouper][algorithm][selector][sampling] = dict()

    results = []
    results.append(pandas.read_csv(str(path.joinpath("run0.csv"))))
    results.append(pandas.read_csv(str(path.joinpath("run1.csv"))))
    results.append(pandas.read_csv(str(path.joinpath("run2.csv"))))
    results.append(pandas.read_csv(str(path.joinpath("run3.csv"))))
    results.append(pandas.read_csv(str(path.joinpath("run4.csv"))))

    recalls = []
    fallouts = []
    precisions = []
    f1s = []
    aucs = []
    accuracies = []

    num_classes = int(results[0].shape[1] / 3)

    for r, run in enumerate(results):
        test_columns = []
        pred_proba_columns = []
        pred_columns = []

        for i in range(num_classes):
            test_columns.append('y_test_%s' % i)
            pred_proba_columns.append('y_pred_proba_%s' % i)
            pred_columns.append('y_pred_%s' % i)

        y_test = run[test_columns].values
        y_pred_proba = run[pred_proba_columns].values
        y_pred = run[pred_columns].values

        y_test_flat = [np.argmax(y) for y in y_test]
        y_pred_flat = [np.argmax(y) for y in y_pred]

        # --- Plot Confusion Matrix ---

        plt.figure(figsize=(10,10))
        cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        with open(str(path.joinpath('group_labels.txt')), 'r') as label:
            labels = label.readlines()
            class_names = [l.split(' - ')[1] for l in labels]
            if grouper == 'CountrySeparatedGroupAndFamiliesGrouper':
                new_names = []
                for class_name in class_names:
                    if '_test' in class_name:
                        new_names.append(class_name.replace('_test\n', '').capitalize())
                class_names = new_names
        df_cm = pandas.DataFrame(cm, index=class_names, columns=class_names)
        sn.heatmap(df_cm, annot=True, fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(str(path.joinpath('confusion_matrix%i.png' % r)))
        plt.close()

        # --- Manual Fallout calculation ---

        class_fallouts = []

        for i in range(num_classes):
            tn, fp, fn, tp = metrics.confusion_matrix(
                [y[i] for y in y_test], [y[i] for y in y_pred]).ravel()
            class_fallouts.append(fp / (fp + tn))

        # ----------------------------------
        if RECALL is True:
            recalls.append(metrics.recall_score(
                y_test_flat, y_pred_flat, average='macro'))

        if FALLOUT is True:
            fallouts.append(np.average(class_fallouts))

        if PRECISION is True:
            precisions.append(metrics.precision_score(
                y_test_flat, y_pred_flat, average='macro'))

        if F1 is True:
            f1s.append(metrics.f1_score(
                y_test_flat, y_pred_flat, average='macro'))

        if AUC is True:
            aucs.append(metrics.roc_auc_score(
                y_test, y_pred_proba, average='macro'))

        if ACCURACY is True:
            accuracies.append(metrics.accuracy_score(y_test_flat, y_pred_flat))

    metric_results = dict()

    if RECALL is True:
        metric_results.update({'recall': {'average': np.average(recalls)}})
        metric_results['recall'].update({'std': float(np.std(recalls))})

    if FALLOUT is True:
        metric_results.update({'fallout': {'average': np.average(fallouts)}})
        metric_results['fallout'].update({'std': float(np.std(fallouts))})

    if PRECISION is True:
        metric_results.update(
            {'precision': {'average': np.average(precisions)}})
        metric_results['precision'].update({'std': float(np.std(precisions))})

    if F1 is True:
        metric_results.update({'f1': {'average': np.average(f1s)}})
        metric_results['f1'].update({'std': float(np.std(f1s))})

    if AUC is True:
        metric_results.update({'auc': {'average': np.average(aucs)}})
        metric_results['auc'].update({'std': float(np.std(aucs))})

    if ACCURACY is True:
        metric_results.update(
            {'accuracy': {'average': np.average(accuracies)}})
        metric_results['accuracy'].update({'std': float(np.std(accuracies))})

    result_dict[grouper][algorithm][selector][sampling] = metric_results

    return result_dict


# -------------------------------------------------------------------------------------------------------------------- #

def print_results_2(table: dict, metric: str):

    print('''
\\begin{tabular}{l|l|r|r|r}
                     & \\textbf{Dataset} & \\textbf{Unbalanced} & \\textbf{Undersampling} & \\textbf{Oversampling} \\\\ \\hline
\multirow{4}{*}{RFC} & Cuckoo & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) \\\\
                     & Cuckoo* & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) \\\\
                     & VirusTotal & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) \\\\
                     & VMRay & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f)\\\\ \\hline
\multirow{4}{*}{DNN} & Cuckoo & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) \\\\
                     & Cuckoo* & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) \\\\
                     & VirusTotal & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) \\\\
                     & VMRay & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f) & %.2f ($\\sigma$: %.2f)\\\\
\\end{tabular}
    ''' % (
        table['RandomForestClassifierAlgorithm']['CuckooFilteredSelector']['Unbalanced'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['CuckooFilteredSelector']['Unbalanced'][metric]['std'],
        table['RandomForestClassifierAlgorithm']['CuckooFilteredSelector']['Undersampling'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['CuckooFilteredSelector']['Undersampling'][metric]['std'],
        table['RandomForestClassifierAlgorithm']['CuckooFilteredSelector']['Oversampling'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['CuckooFilteredSelector']['Oversampling'][metric]['std'],
		
		table['RandomForestClassifierAlgorithm']['CuckooExtractedSelector']['Unbalanced'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['CuckooExtractedSelector']['Unbalanced'][metric]['std'],
        table['RandomForestClassifierAlgorithm']['CuckooExtractedSelector']['Undersampling'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['CuckooExtractedSelector']['Undersampling'][metric]['std'],
        table['RandomForestClassifierAlgorithm']['CuckooExtractedSelector']['Oversampling'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['CuckooExtractedSelector']['Oversampling'][metric]['std'],

        table['RandomForestClassifierAlgorithm']['VirusTotalCleanSelector']['Unbalanced'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['VirusTotalCleanSelector']['Unbalanced'][metric]['std'],
        table['RandomForestClassifierAlgorithm']['VirusTotalCleanSelector']['Undersampling'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['VirusTotalCleanSelector']['Undersampling'][metric]['std'],
        table['RandomForestClassifierAlgorithm']['VirusTotalCleanSelector']['Oversampling'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['VirusTotalCleanSelector']['Oversampling'][metric]['std'],

        table['RandomForestClassifierAlgorithm']['VMRayCleanSelector']['Unbalanced'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['VMRayCleanSelector']['Unbalanced'][metric]['std'],
        table['RandomForestClassifierAlgorithm']['VMRayCleanSelector']['Undersampling'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['VMRayCleanSelector']['Undersampling'][metric]['std'],
        table['RandomForestClassifierAlgorithm']['VMRayCleanSelector']['Oversampling'][metric]['average'],
        table['RandomForestClassifierAlgorithm']['VMRayCleanSelector']['Oversampling'][metric]['std'],
		
		table['NeuralNetworkAlgorithm']['CuckooFilteredSelector']['Unbalanced'][metric]['average'],
        table['NeuralNetworkAlgorithm']['CuckooFilteredSelector']['Unbalanced'][metric]['std'],
        table['NeuralNetworkAlgorithm']['CuckooFilteredSelector']['Undersampling'][metric]['average'],
        table['NeuralNetworkAlgorithm']['CuckooFilteredSelector']['Undersampling'][metric]['std'],
        table['NeuralNetworkAlgorithm']['CuckooFilteredSelector']['Oversampling'][metric]['average'],
        table['NeuralNetworkAlgorithm']['CuckooFilteredSelector']['Oversampling'][metric]['std'],
		
		table['NeuralNetworkAlgorithm']['CuckooExtractedSelector']['Unbalanced'][metric]['average'],
        table['NeuralNetworkAlgorithm']['CuckooExtractedSelector']['Unbalanced'][metric]['std'],
        table['NeuralNetworkAlgorithm']['CuckooExtractedSelector']['Undersampling'][metric]['average'],
        table['NeuralNetworkAlgorithm']['CuckooExtractedSelector']['Undersampling'][metric]['std'],
        table['NeuralNetworkAlgorithm']['CuckooExtractedSelector']['Oversampling'][metric]['average'],
        table['NeuralNetworkAlgorithm']['CuckooExtractedSelector']['Oversampling'][metric]['std'],

        table['NeuralNetworkAlgorithm']['VirusTotalCleanSelector']['Unbalanced'][metric]['average'],
        table['NeuralNetworkAlgorithm']['VirusTotalCleanSelector']['Unbalanced'][metric]['std'],
        table['NeuralNetworkAlgorithm']['VirusTotalCleanSelector']['Undersampling'][metric]['average'],
        table['NeuralNetworkAlgorithm']['VirusTotalCleanSelector']['Undersampling'][metric]['std'],
        table['NeuralNetworkAlgorithm']['VirusTotalCleanSelector']['Oversampling'][metric]['average'],
        table['NeuralNetworkAlgorithm']['VirusTotalCleanSelector']['Oversampling'][metric]['std'],

        table['NeuralNetworkAlgorithm']['VMRayCleanSelector']['Unbalanced'][metric]['average'],
        table['NeuralNetworkAlgorithm']['VMRayCleanSelector']['Unbalanced'][metric]['std'],
        table['NeuralNetworkAlgorithm']['VMRayCleanSelector']['Undersampling'][metric]['average'],
        table['NeuralNetworkAlgorithm']['VMRayCleanSelector']['Undersampling'][metric]['std'],
        table['NeuralNetworkAlgorithm']['VMRayCleanSelector']['Oversampling'][metric]['average'],
        table['NeuralNetworkAlgorithm']['VMRayCleanSelector']['Oversampling'][metric]['std']
    )
    )


def load_data(in_path):
    table_data = dict()
    for (dirpath, dirnames, _) in os.walk(str(in_path)):
        for dirname in dirnames:
            table_data = calc_metrics(
                Path(dirpath).joinpath(dirname), table_data)
	
    return table_data


def print_tables(table_data):
    print('APTGrouper')
    print_results_2(table_data['APTGrouper'], 'accuracy')
    print()
	  
    print('CountryGrouper')
    print_results_2(table_data['CountryGrouper'], 'accuracy')
    print()
    
    print('CountrySeparatedGroupAndFamiliesGrouper')
    print_results_2(table_data['CountrySeparatedGroupAndFamiliesGrouper'], 'accuracy')
    print()
    

if __name__ == "__main__":
    table_data = load_data(OUT_PATH)
    print_tables(table_data)
    with open(str(CURRENT_PATH.joinpath('results.json')), 'w') as f:
        json.dump(table_data, f)

