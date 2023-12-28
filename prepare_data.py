import sys
import json
import os
import csv
import spacy
import argparse
import pandas as pd
from tqdm import tqdm


class FilterCorpus:
    """
    Filtering a big corpus to smaller corpus, according to the chosen fields
    """

    def __init__(self, to_filter, filtered_json):
        self.json_to_filter = to_filter
        self.filtered_json = filtered_json
        self.writing_file = None
        self.lines = set()

    def read_write_json(self):
        """
        generator for reading the json file
        """
        with open(self.json_to_filter, encoding='utf-8', errors='ignore') as outfile:
            with open(self.filtered_json, 'w') as infile:
                self.writing_file = infile
                for line in outfile:
                    yield json.loads(line)

    def update_fields(self, fields_lst, lines):
        """
        filter the lines by their fieldsOfStudy - writing to json only the relevant articles
        """
        filtered_line = False
        for line in tqdm(lines):
            for field in fields_lst:
                field = field.replace(",", " ")
                if field in line['fieldsOfStudy']:
                    filtered_line = True
            if filtered_line:
                json.dump(line, self.writing_file)
                self.writing_file.write("\n")
            filtered_line = False


class DataPreProcessing:

    def __init__(self, read_from, write_to):
        self.read_from = read_from
        self.write_to = write_to
        self.nlp = spacy.load('en')
        self.nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    def read_json(self):
        """
        reading the jsonl file
        """
        with open(self.read_from, encoding='utf-8', errors='ignore') as outfile:
            for line in outfile:
                yield json.loads(line)

    def filter_english(self, lines):
        """
        filtering the English papers
        """
        for line in tqdm(lines):
            title = line["title"]
            doc = self.nlp(title)
            if doc._.language["language"] == 'en':
                yield line

    def filter_not_english(self, lines):
        """
        filtering the non-English papers
        """
        for line in tqdm(lines):
            abstract = line["paperAbstract"]
            doc = self.nlp(abstract)
            if doc._.language["language"] != "en":
                yield line

    def write_to_json(self, lines):
        """
        write to jsonl file the English filtered lines
        """
        with open(self.write_to, 'w') as f:
            for line in tqdm(lines):
                f.write(json.dumps(line) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_name', help='the corpus to filter')
    parser.add_argument('csv_file', help='the name of the CSV file')
    parser.add_argument('command', help='command')
    args = parser.parse_args()
    data_pre = DataPreProcessing(args.corpus_name, args.csv_file)
    return data_pre, args.csv_file, args.command


if __name__ == '__main__':
    data_pre_processing, csv_name, command = parse_args()

    # for filtering the English papers
    if command == 'filter_english':
        all_lines = data_pre_processing.read_json()
        all_english_lines = data_pre_processing.filter_english(all_lines)
        data_pre_processing.write_to_json(all_english_lines)

    # for filtering the non-English papers
    elif command == 'filter_not_english':
        all_lines = data_pre_processing.read_json()
        all_english_lines = data_pre_processing.filter_not_english(all_lines)
        data_pre_processing.write_to_json(all_english_lines)


def gen(file_name):
    """
    generator for reading the json file
    """
    with open(file_name, encoding='utf-8', errors='ignore') as json_file:
        for line in json_file:
            yield json.loads(line)


# Dictionary in the format of {field_name: number of articles}
fields_dict = {}


def update_fields(lines):
    """
    update the fields dictionary
    """
    for line in tqdm(lines):
        for field in line['fieldsOfStudy']:
            if field not in fields_dict:
                fields_dict.update({field: 1})
            else:
                fields_dict[field] += 1


def write_to_csv(csv_file):
    """
    write the dictionary to csv file
    """
    table_cols = list(fields_dict.keys())

    with open(csv_file, 'w', newline='') as csv_file:
        """
        Writing dictionary to csv file
        """
        writer = csv.DictWriter(csv_file, fieldnames=table_cols)
        writer.writeheader()
        for data in [fields_dict]:
            writer.writerow(data)


# For running: python3 filter_articles_to_fields.py corpus_name csv_name
# The program creates a csv file with all the fieldsOfStudy and the number of the articles belonging to each field
if __name__ == '__main__':
    corpus_name = sys.argv[1]
    csv_name = sys.argv[2]
    all_lines = gen(corpus_name)
    update_fields(all_lines)
    write_to_csv(csv_name)
