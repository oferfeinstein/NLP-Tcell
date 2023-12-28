import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from simpletransformers.classification import MultiLabelClassificationModel

class PredictModel:
    def __init__(self):
        self.labels = []
        self.predicts = []
        self.labels = []
        self.predictions = []

    def get_test_text(self):
        """
        getting the test set (the test abstract)
        """
        df = pd.read_csv('test.csv', sep='\t')
        for text in tqdm(df['text']):
            self.predicts.append(str(text))
        return self.predicts


    def calculate_reports(self):
        """
        calculating the precision after performing prediction by the model, but only with the relevant or not-relevant
        label
        """
        df = pd.read_csv('test.csv', converters={'labels': convert_str_to_list, 'predictions': convert_str_to_list})

        # removing nan values
        labels = df['labels'].dropna()
        predictions = df['predictions'].dropna()

        labels_arr = np.array(labels)
        last_label = []
        for lst in labels_arr:
            if len(lst) != 0:
                last_label.append(lst[19])
        preds_arr = np.array(predictions)
        last_pred = []
        for lst in preds_arr:
            if len(lst) != 0:
                last_pred.append(lst[19])

        cf_matrix = skm.confusion_matrix(last_label[:-1], last_pred)

        # classification report only for the last label
        classification_report = skm.classification_report(last_label[:-1], last_pred)

        return cf_matrix, classification_report



def add_pred_to_csv(predictions_vec):
    """
    adding to csv the model predictions vectors
    """
    df = pd.read_csv('test.csv', sep='\t')
    df['predictions'] = predictions_vec
    df.to_csv('test.csv', index=False)


def convert_str_to_list(str_labels):
    labels_lst = []
    for label in str_labels:
        if label == "1":
            labels_lst.append(1)
        elif label == "0":
            labels_lst.append(0)
    return labels_lst


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            #stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                #accuracy, precision, recall, f1_score)
            stats_text = "\nPrecision={:0.3f}".format(precision)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.savefig("cf")


if __name__ == '__main__':
    pred = PredictModel()

    # prediction
    model = MultiLabelClassificationModel('bert', model_name='/cs/labs/ravehb/ofer.feinstein/project/outputs_hidden/', cuda_device=1)
    pred_text = pred.get_test_text()
    predictions, raw_outputs = model.predict(pred_text)
    add_pred_to_csv(predictions)

    # after labels prediction - calculating the precision but only with the last label (if the paper is relevant or not)
    cf_matrix, classification_report = pred.calculate_reports()

    make_confusion_matrix(cf_matrix,
                          group_names=["True Negative", "False Positive", "False Negative", "True Positive"],
                          categories='auto',
                          count=True,
                          percent=False,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=(10,8),
                          cmap='BuPu',
                          title='Confusion matrix for the relevant label')
