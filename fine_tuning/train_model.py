import argparse
import csv
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from simpletransformers.classification import MultiLabelClassificationModel

COLS = ["text"]
LABELS_DICT = {
    "Chemistry": 0, "Biology": 1, "Medicine": 2, "Computer Science": 3,
    "Art": 4, "Physics": 5, "Engineering": 6, "History": 7, "Geography": 8, "Psychology": 9,
    "Business": 10, "Geology": 11, "Economics": 12, "Political Science": 13,
    "Environmental Science": 14, "Mathematics": 15, "Materials Science": 16, "Philosophy": 17,
    "Sociology": 18
}


class TrainModel:
    def __init__(self, read_from, write_to):
        self.labels = []
        self.read_from = read_from
        self.predictions = []
        self.write_to = write_to

    def read_json(self):
        """
        Reading the jsonl file
        """
        with open(self.read_from, encoding='utf-8', errors='ignore') as outfile:
            for line in outfile:
                yield json.loads(line)

    def write_labels(self, new_line, is_relevant=True):
        """
        Adding the labels column
        """
        labels_vec = [0] * 20
        fields_lst = new_line["fieldsOfStudy"]
        for field in fields_lst:
            labels_vec[LABELS_DICT[field]] = 1
        if is_relevant:
            labels_vec[19] = 1
        self.labels.append(labels_vec)

    def write_to_csv(self, lines):
        """
        Writing to CSV in BERT input format
        """
        with open(self.write_to, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(COLS)
            for line in tqdm(lines):
                csv_line = [line["paperAbstract"]]
                writer.writerow(csv_line)

    def add_labels_to_df(self, is_relevant=True):
        """
        Adding the labels to the data frame. Notice that the CSV to read is not a parameter.
        """
        if is_relevant:
            df = pd.read_csv('cp_filter_english.csv')
        else:
            df = pd.read_csv('cp_non_relevant.csv', sep='\t')
        df['labels'] = self.labels
        return df


def write_df_to_csv(file_name, df):
    """
    Writing the data frames to a CSV file
    """
    df.to_csv(file_name, sep='\t', encoding='utf-8')


def parse_args():
    """
    Parsing the input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('relevant_corpus', help='Corpus with relevant papers in jsonl format')
    parser.add_argument('non_relevant_corpus', help='Corpus with not relevant papers in jsonl format')
    parser.add_argument('relevant_csv', help='Name for the new CSV created from the jsonl lines')
    parser.add_argument('not_relevant_csv', help='Name for the new CSV created from the jsonl lines')
    args = parser.parse_args()
    return args.relevant_corpus, args.non_relevant_corpus, args.relevant_csv, args.not_relevant_csv


if __name__ == '__main__':
    relevant_corpus, non_relevant_corpus, relevant_csv, non_relevant_csv = parse_args()

    relevant = TrainModel(relevant_corpus, relevant_csv)
    non_relevant = TrainModel(non_relevant_corpus, non_relevant_csv)

    all_relevant_lines = relevant.read_json()
    all_non_relevant_lines = non_relevant.read_json()

    # Write the jsonl to CSV
    relevant.write_to_csv(all_relevant_lines)
    non_relevant.write_to_csv(all_non_relevant_lines)

    all_relevant_lines = relevant.read_json()
    all_non_relevant_lines = non_relevant.read_json()

    # Writing the labels for the relevant and not relevant papers
    for line in tqdm(all_relevant_lines):
        relevant.write_labels(line)
    relevant_df = relevant.add_labels_to_df()

    for line in tqdm(all_non_relevant_lines):
        non_relevant.write_labels(line, False)
    non_relevant_df = non_relevant.add_labels_to_df(False)

    # Combining the two data frames into one
    new_corpus = pd.concat([relevant_df, non_relevant_df])

    write_df_to_csv('cp_new_corpus.csv', new_corpus)

    # Splitting the data into train, test & validation
    train_df, evaltest_df = train_test_split(new_corpus, test_size=0.2, random_state=1)
    write_df_to_csv('train.csv', train_df)

    eval_df, test_df = train_test_split(evaltest_df, test_size=0.5, random_state=1)
    write_df_to_csv('eval.csv', eval_df)
    write_df_to_csv('test.csv', test_df)

    # Training the model
    model = MultiLabelClassificationModel('bert', 'bert-base-cased',
                                          num_labels=20,
                                          args={'train_batch_size': 4, 'gradient_accumulation_steps': 16,
                                                'learning_rate': 3e-5, 'num_train_epochs': 3,
                                                'max_seq_length': 512, 'fp16': False})
    model.train_model(train_df)
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
