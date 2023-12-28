import argparse
import json
from tqdm import tqdm
import allennlp_models.tagging
from allennlp.predictors.predictor import Predictor

class OIEModel:
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name

    def read_json(self):
        """
        Read the JSONL file and yield parsed lines
        """
        with open(self.corpus_name, encoding='utf-8', errors='ignore') as outfile:
            for line in outfile:
                yield json.loads(line)

    def get_parsed_abstract(self, lines):
        """
        Get parsed abstracts from lines
        """
        for line in tqdm(lines):
            yield line['parsed abstract']

def run_model(parsed_abstract_lines):
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", cuda_device=1)

    sentences_lst = []
    for abstract in tqdm(parsed_abstract_lines):
        for sentence in abstract:
            sentences_lst.append({'sentence': sentence})

    i = 0
    while i < len(sentences_lst):
        print(i)
        oie_lines = predictor.predict_batch_json(sentences_lst[i:i + 200])
        with open('oie_lines_new.jsonl', 'a') as output_file:
            for line in oie_lines:
                output_file.write(json.dumps(line) + '\n')
        i += 200

    oie_lines = predictor.predict_batch_json(sentences_lst[i:len(sentences_lst)])
    with open('oie_lines_new.jsonl', 'a') as output_file:
        for line in oie_lines:
            output_file.write(json.dumps(line) + '\n')

def get_args():
    """
    Get arguments from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_name')
    args = parser.parse_args()
    return args.corpus_name

if __name__ == '__main__':
    corpus_name = get_args()
    model = OIEModel(corpus_name)
    lines = model.read_json()
    parsed_abstracts = model.get_parsed_abstract(lines)
    run_model(parsed_abstracts)
