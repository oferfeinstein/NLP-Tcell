import argparse
import json
from tqdm import tqdm


def read_json(jsonl):
    """
    reading the jsonl file
    """
    with open(jsonl, encoding='utf-8', errors='ignore') as outfile:
        for line in outfile:
            yield json.loads(line)


def create_new_corpus(big_line, small_line, new_corpus):
    """
    adding to the new corpus only lines that don't appear in the 'filter_english' corpus
    """
    with open(new_corpus, 'a') as f:
        if big_line['id'] != small_line['id']:
            f.write(json.dumps(big_line) + "\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('big_corpus_name', help='the corpus to filter')
    parser.add_argument('small_corpus_name', help='the corpus to filter')
    parser.add_argument('new_corpus', help='command')
    args = parser.parse_args()
    return args.big_corpus_name, args.small_corpus_name, args.new_corpus


if __name__ == '__main__':
    big_corpus, small_corpus, new_corpus = parse_args()
    big_all_lines = read_json(big_corpus)
    small_all_lines = read_json(small_corpus)
    i = 0
    for small_line in small_all_lines:
        for big_line in big_all_lines:
            if i != 524904:
                create_new_corpus(big_line, small_line, new_corpus)
                i += 1
            else:
                break
