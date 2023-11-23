from common import *
from pyspark.sql import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt 
import numpy as np 

def eval_predictions(predictions: DataFrame) -> None:    

    labels = predictions.select(COL_LABEL).to_koalas().to_numpy()
    predictions = predictions.select(COL_PREDICTION).to_koalas().to_numpy()

    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = accuracy_score(labels, predictions)    

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1_score}')

def plot_confusion_matrix(predictions_df: DataFrame,
                          normalize: bool = False,
                          title: str = None,
                          save_fig: bool = False,
                          save_path: str = None):
    '''
    Plots a Confusion Matrix computed on the given `DataFrame`
    
    Args:
        - predictions_df: a `DataFrame` that has at least a `TARGET_VARIABLE_COL` and a `PREDICTED_TARGET_VARIABLE_COL`
        - normalize: True to normalize the values in the Confusion Matrix, False otherwise
        - title: optional title to append on top of the plotted Confusion Matrix
        - save_fig: True to save the confusion matrix figure (default: False)
        - save_path: path to save the confusion matrix figure (default: None)

    '''
    
    labels = predictions_df.select(COL_WEATHER_CONDITION).to_koalas().to_numpy()
    predictions = predictions_df.select(COL_PREDICTED_TARGET_VARIABLE).to_koalas().to_numpy()

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions, labels=None)
    
    # Only use the labels that appear in the data
    classes = unique_labels(labels, predictions)

    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,           
           xlabel='Predicted label', ylabel='True label',
           title=title)

    ax.set_ylim(len(classes) - 0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    if save_fig and save_path is not None:
        plt.axes("off")
        plt.savefig(save_path)
    elif save_path is None:
        raise ValueError("Save path is not specified")

    fig.tight_layout()
    plt.show()