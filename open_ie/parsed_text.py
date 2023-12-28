import json
import spacy
import argparse
from tqdm import tqdm
from pysbd.utils import PySBDFactory

class ParsedAbstract:
    def __init__(self, read_from, write_to):
        self.read_from = read_from
        self.write_to = write_to
        self.writing_file = None
        self.parsed_abstract = []
        self.nlp = spacy.blank('en')
        self.nlp.add_pipe(PySBDFactory(self.nlp))

    def read_write_json(self):
        """
        Read the JSONL file and open a new file for writing
        """
        with open(self.read_from, encoding='utf-8', errors='ignore') as outfile:
            with open(self.write_to, 'w') as infile:
                self.writing_file = infile
                for line in outfile:
                    yield json.loads(line)

    def add_data_to_line(self, lines):
        """
        Add parsed abstracts to each line
        """
        for line in tqdm(lines):
            abstract = self.split_to_sentences(line["paperAbstract"])
            line['parsed abstract'] = abstract
            yield line

    def split_to_sentences(self, abstract):
        """
        Split abstract into sentences
        """
        doc = self.nlp(abstract)
        lst_abstract = []
        for sent in doc.sents:
            sents_lst = [str(word) for word in sent]
            sents_str = " ".join(sents_lst)
            lst_abstract.append(sents_str)
        return lst_abstract

    def write_to_jsonl(self, lines):
        """
        Write modified lines to the output file
        """
        for line in lines:
            json.dump(line, self.writing_file)
            self.writing_file.write("\n")

    def join_abstract(self, line):
        parsed_abstract = line["parsed abstract"]
        join_abstract = "\n".join(parsed_abstract)
        text_label = {"text": join_abstract, "labels": []}
        yield text_label

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_name', help='The corpus to filter')
    parser.add_argument('file_to_write', help='The file to write')
    args = parser.parse_args()
    parser = ParsedAbstract(args.corpus_name, args.file_to_write)
    return parser

if __name__ == '__main__':
    # Create the parsed abstract
    parser = parse_args()
    all_lines = parser.read_write_json()
    update_lines = parser.add_data_to_line(all_lines)
    parser.write_to_jsonl(update_lines)
