from helper import *
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from pathlib import Path
import shutil


def append_secondary_features(files: [Path], bag_of_words):
    secondary_features = {
        '"function_name":"IsNetworkAlive"': 'APT 1',
        '"regkey_r":"Software\\\\Microsoft\\\\AdvancedINFSetup"': 'APT 10',
        '"filepath": "C:\\Users\\John\\AppData\\Local\\Temp\\msi.dll"': 'APT 19',
        '"filepath":"C:\\Windows\\System32\\FastUserSwitchingCompatibilityex.dll"': 'APT 21',
        '"function_name":"_snwprintf"': 'APT 28',
        '"function_name": "CoInternetIsFeatureEnabledForUrl"': 'APT 29',
        '"mutant_name": "Microsoft': 'APT 30',
        '"process_name": "igfxext.exe"': 'Dark Hotel',
        '"regkey_r": "fertger"': 'Energetic Bear',
        '"newfilepath":"C:\\fanny.bmp"': 'Equation Group',
        '"filepath":"C:\\ProgramFiles\\CommonFiles\\MicrosoftShared\\OFFICE14\\1033\\ALRTINTL.DLL"': 'Gorgon Group',
        '"filepath":"C:\\Windows\\winmm.dll"': 'Winnti'
    }

    appendix = []

    for i in files:
        path_parts = Path(i).parts
        argument_file = report_path.joinpath(path_parts[-2], 'cuckoo_extracted_arguments', path_parts[-1])
        content = open(str(argument_file), 'r').read()
        secondary_features_result = []

        for j in secondary_features.keys():
            if j in content:
                secondary_features_result.append(1)
            else:
                secondary_features_result.append(0)

        appendix.append(secondary_features_result)

    bag_of_words_t = bag_of_words.transpose()
    appendix_t = sparse.csr_matrix(np.array(appendix)).transpose()

    combined = csr_vappend(bag_of_words_t, appendix_t)
    return sparse.csr_matrix(combined).transpose()


def create_bag_of_words(x_train: [Path], y_train: [int], x_test: [Path], y_test: [int], additional_files: bool):
    print('Start converting to bag of words...', end='', flush=True)

    vectorizer = CountVectorizer(input='filename', max_features=50000, max_df=len(y_train) + len(y_test) - 1)
    bag_of_words = vectorizer.fit_transform(map(str, x_train + x_test))

    if additional_files:
        bag_of_words = append_secondary_features(x_train + x_test, bag_of_words)

    # Clean up directory (delete & recreate)
    shutil.rmtree(str(result_path))
    os.mkdir(str(result_path))

    if len(y_test) > 0:
        sparse.save_npz(result_path.joinpath("bag_of_words_train.npz"), bag_of_words[:len(y_train)])
        with open(str(result_path.joinpath('train_labels.txt')), 'w') as wf:
            wf.write('\n'.join(map(str, y_train)))

        sparse.save_npz(result_path.joinpath("bag_of_words_test.npz"),
                        bag_of_words[len(y_train):len(y_train) + len(y_test)])
        with open(str(result_path.joinpath('test_labels.txt')), 'w') as wf:
            wf.write('\n'.join(map(str, y_test)))
    else:
        sparse.save_npz(result_path.joinpath("bag_of_words.npz"), bag_of_words)
        with open(str(result_path.joinpath('labels.txt')), 'w') as wf:
            wf.write('\n'.join(map(str, y_train)))

    with open(str(result_path.joinpath('words.txt')), 'w') as wf:
        wf.write('\n'.join(map(str, vectorizer.get_feature_names())))

    print('done!')
