import csv
import configparser
import json
import os
import pprint
import yaml

from typing import Mapping

# Reader
def read_yaml(filename: str) -> Mapping :
    data = {}
    try:
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(e)
    pprint.pprint(data)
    return data

def read_json(filename: str) -> Mapping :
    data = {}
    with open(filename, "r", encoding="utf8") as f:
        data = json.load(f)

    # pprint.pprint(data)
    return data

# Writer
def write_yaml(filename: str, data: Mapping):
    pass

def write_json(filename: str, data: Mapping):
    with open(filename, "w", encoding="utf8") as fw:
        jsonString = json.dumps(data, ensure_ascii=False)
        fw.writelines(jsonString)

def write_csv(filename: str, data: list):
    with open(f'{filename}.csv', 'w', newline='', encoding="utf8") as csvfile:
        if len(data) == 0:
            return
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def read_ini(filename: str) -> Mapping:
    config = configparser.ConfigParser()
    config.read(filename)

    return config