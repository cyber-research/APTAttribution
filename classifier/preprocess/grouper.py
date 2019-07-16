from abc import ABCMeta, abstractmethod
from helper import *


class Grouper(metaclass=ABCMeta):
    @abstractmethod
    def process_sample(self, sample: (str, Path), words_in_sample: [str], data_set_stats: pandas.DataFrame):
        pass

    @abstractmethod
    def release_results(self) -> [([Path], [int])]:
        pass


class CountrySeparatedGroupAndFamiliesGrouper(Grouper):

    category_counts = {}
    result_files = []

    def process_sample(self, sample: (str, Path), words_in_sample: [str], data_set_stats: pandas.DataFrame):

        apt = data_set_stats.loc[sample[0]]['APT-group']

        if apt in ['APT 1', 'APT 10', 'Winnti']:
            category = 'china_train'
        elif apt in ['APT 19', 'APT 21', 'APT 30']:
            category = 'china_test'
        elif apt in ['APT 28', 'APT 29']:
            category = 'russia_train'
        elif apt in ['Energetic Bear']:
            category = 'russia_test'
        else:  # Family not in scope for this test
            return

        if not os.path.exists(text_reports_path.joinpath(apt)):
            os.makedirs(str(text_reports_path.joinpath(apt)))

        if category not in self.category_counts.keys():
            self.category_counts[category] = 0

        self.category_counts[category] += 1

        file_path = text_reports_path.joinpath(apt, sample[0] + '.txt')
        self.result_files.append((category, file_path))

        # Perhaps keep words as they are, e.g. treat 'Mon 26 Oct' as one word.
        with open(str(file_path), 'w') as wf:
            wf.write('\n'.join(map(str, words_in_sample)))

    def release_results(self):
        x_train = [rf[1] for rf in self.result_files if 'train' in rf[0]]
        x_test = [rf[1] for rf in self.result_files if 'test' in rf[0]]
        y_train = [int('russia' in rf[0]) for rf in self.result_files if 'train' in rf[0]]
        y_test = [int('russia' in rf[0]) for rf in self.result_files if 'test' in rf[0]]

        # Write category numbers and labels
        with open(str(temp_path.joinpath('group_labels.txt')), 'w') as wf:
            wf.write('\n'.join(['%s - %s' % (i, key,) for i, key in enumerate(self.category_counts.keys())]))

        return [(x_train, y_train), (x_test, y_test)]


class SimpleGrouper(Grouper):

    key = ''
    category_counts = {}
    result_files = []

    def setkey(self, val):
        self.key = val

    @abstractmethod
    def process_sample(self, sample: (str, Path), words_in_sample: [str], data_set_stats: pandas.DataFrame):
        category = data_set_stats.loc[sample[0]][self.key]
        apt = data_set_stats.loc[sample[0]]['APT-group']

        if category not in self.category_counts.keys():
            self.category_counts[category] = 0

        self.category_counts[category] += 1

        if not os.path.exists(text_reports_path.joinpath(apt)):
            os.makedirs(str(text_reports_path.joinpath(apt)))

        file_path = text_reports_path.joinpath(apt, sample[0] + '.txt')
        self.result_files.append((category, file_path))

        # Perhaps keep words as they are, e.g. treat 'Mon 26 Oct' as one word.
        with open(str(file_path), 'w') as wf:
            wf.write('\n'.join(map(str, words_in_sample)))

    @abstractmethod
    def release_results(self) -> [([Path], [int])]:
        x = [rf[1] for rf in self.result_files]
        y = [list(self.category_counts.keys()).index(rf[0]) for rf in self.result_files]

        # Write category numbers and labels
        with open(str(temp_path.joinpath('group_labels.txt')), 'w') as wf:
            wf.write('\n'.join(['%s - %s' % (i, key,) for i, key in enumerate(self.category_counts.keys())]))

        return [(x, y), ([], [])]


class CountryGrouper(SimpleGrouper):

    def process_sample(self, sample: (str, Path), words_in_sample: [str], data_set_stats: pandas.DataFrame):
        super().setkey('Country')
        super().process_sample(sample, words_in_sample, data_set_stats)

    def release_results(self):
        return super().release_results()


class APTGrouper(SimpleGrouper):

    def process_sample(self, sample: (str, Path), words_in_sample: [str], data_set_stats: pandas.DataFrame):
        super().setkey('APT-group')
        super().process_sample(sample, words_in_sample, data_set_stats)

    def release_results(self):
        return super().release_results()


class FamilyGrouper(SimpleGrouper):

    def process_sample(self, sample: (str, Path), words_in_sample: [str], data_set_stats: pandas.DataFrame):
        super().setkey('Family')
        super().process_sample(sample, words_in_sample, data_set_stats)

    def release_results(self):
        return super().release_results()


def group_switch(name: str) -> Grouper:
    switcher = {
        'CountrySeparatedGroupAndFamiliesGrouper': CountrySeparatedGroupAndFamiliesGrouper,
        'CountryGrouper': CountryGrouper,
        'APTGrouper': APTGrouper,
        'FamilyGrouper': FamilyGrouper
    }

    group = switcher.get(name, lambda: "Selector not found!")
    return group()
