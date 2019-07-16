import json

from grouper import *
from helper import *


class Extractor(metaclass=ABCMeta):
    @abstractmethod
    def extract(self, group_instance: Grouper, reports_to_read: [(str, Path)], data_set_stats: pandas.DataFrame):
        pass


class JSONExtractor(Extractor):
    def extract(self, group_instance: Grouper, reports_to_read: [(str, Path)], data_set_stats: pandas.DataFrame):

        words_in_reports = []
        i = 0

        while i < len(reports_to_read):

            # If new sample is encountered, write away the info for the previous and clear words-array
            if i != 0 and reports_to_read[i][0] != reports_to_read[i - 1][0]:
                group_instance.process_sample(reports_to_read[i - 1], words_in_reports, data_set_stats)
                words_in_reports = []

            with open(str(reports_to_read[i][1]), 'r') as rf:
                # Read report as JSON
                file_content = rf.read()
                parsed_report = json.loads(file_content)
                json_extract_words(parsed_report, words_in_reports)

            i += 1

        # Process last sample
        group_instance.process_sample(reports_to_read[i - 1], words_in_reports, data_set_stats)


class TextExtractor(Extractor):
    def extract(self, group_instance: Grouper, reports_to_read: [(str, Path)], data_set_stats: pandas.DataFrame):

        for report in reports_to_read:
            with open(str(report[1]), 'r') as rf:
                group_instance.process_sample(report, [x.replace('\n', '').replace('"', '') for x in rf.readlines()],
                                              data_set_stats)
