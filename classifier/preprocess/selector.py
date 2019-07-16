import re

from extractor import *


class Selector(metaclass=ABCMeta):
    @abstractmethod
    def select(self) -> [(str, Path)]:
        pass


class CuckooSelector(Selector):
    def select(self):
        reports = []
        for (dirpath, dirnames, filenames) in os.walk(str(report_path)):
            for filename in filenames:
                if Path(filename).suffix == '.json' and 'cuckoo' in dirpath:
                    reports.append((re.split('[_ .]', filename)[0], Path(dirpath).joinpath(filename)))

        return reports


class CuckooFilteredSelector(Selector):
    def select(self):
        reports = []
        for (dirpath, dirnames, filenames) in os.walk(str(report_path)):
            for filename in filenames:
                if Path(filename).suffix == '.json' and 'cuckoo_filtered' in dirpath:
                    reports.append((re.split('[_ .]', filename)[0], Path(dirpath).joinpath(filename)))

        return reports


class CuckooExtractedSelector(Selector):
    def select(self):
        reports = []
        for (dirpath, dirnames, filenames) in os.walk(str(report_path)):
            for filename in filenames:
                if Path(filename).suffix == '.txt' and 'cuckoo_extracted_api' in dirpath:
                        reports.append((re.split('[_ .]', filename)[0], Path(dirpath).joinpath(filename)))

        return reports


def selector_switch(name: str) -> (Selector, Extractor):
    selector_switcher = {
        'CuckooSelector': CuckooSelector,
        'CuckooFilteredSelector': CuckooFilteredSelector,
        'CuckooExtractedSelector': CuckooExtractedSelector,
    }

    extractor_switcher = {
        'CuckooSelector': JSONExtractor,
        'CuckooFilteredSelector': JSONExtractor,
        'CuckooExtractedSelector': TextExtractor,
    }

    selector_instance = selector_switcher.get(name, lambda: "Selector not found!")
    extractor_instance = extractor_switcher.get(name, lambda: "Selector not found!")
    return selector_instance(), extractor_instance()
