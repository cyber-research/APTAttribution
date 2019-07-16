from pathlib import Path
import os
import requests
import json

current_directory = Path(os.path.abspath(__file__)).parent

def perform_report_request(filename, task_id):
    BASE_URL = 'https://cuckoo.local' 
    url = BASE_URL + '/api/tasks/report/%s' % task_id
    filepath = Path(filename)
    r = requests.get(url)
    if r.status_code == requests.codes.ok:
        report = r.json()
        new_dir = current_directory.parent.parent.joinpath('reports', filepath.parent)
        if not os.path.isdir(str(new_dir)):
            os.mkdir(str(new_dir))
        new_dir = new_dir.joinpath('cuckoo')
        if not os.path.isdir(str(new_dir)):
            os.mkdir(str(new_dir))
        new_file = str(new_dir.joinpath(filepath.name + '.json'))
        with(open(new_file, 'w+')) as j:
            json.dump(report, j)

with open(str(current_directory.joinpath('succeeded.txt')), 'r') as f:
    submissions_to_retrieve = [(s.split(' - ')[0], s.split(' - ')[1]) for s in f]

i = 1

for s in submissions_to_retrieve:
    perform_report_request(s[0], s[1])
    print(i, s)
    i += 1
