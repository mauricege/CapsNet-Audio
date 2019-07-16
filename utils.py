import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import pandas
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix
from typing import List, Tuple


def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4, 6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def _transform_arrays(y_true: np.array,
                      y_pred: np.array) -> (np.array, np.array):

    if y_true.shape[1] > 1:
        y_true_transformed = np.argmax(y_true, axis=1)
    if y_pred.shape[1] > 1:
        y_pred_transformed = np.argmax(y_pred, axis=1)
    assert (
        y_true.shape == y_pred.shape
    ), f'Shapes of predictions and labels for multiclass classification should conform to (n_samples,) but received {y_pred.shape} and {y_true.shape}.'
    return y_true_transformed, y_pred_transformed


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num) / width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num) / height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


class MetricCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 labels: List,
                 validation_data: Tuple[np.array],
                 batch_size=10):
        super().__init__()
        self.labels = {index: name for index, name in enumerate(labels)}
        self.validation_data = validation_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1][0]
        y_pred = np.asarray(
            self.model.predict(X_val, batch_size=self.batch_size)[0])
        recall = recall_score(y_val, y_pred, average='macro')
        precision = precision_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')

        print(f'UAR: {recall}')
        self._data.append({
            'val_rec_macro': recall,
        })
        logs['vall_rec_macro'] = recall

        print(f'Macro Precision: {precision}')
        self._data[-1]['val_prec_macro'] = precision
        logs['vall_prec_macro'] = precision

        print(f'Macro F1: {f1}')
        self._data[-1]['val_f1_macro'] = f1
        logs['vall_f1_macro'] = f1

        y_val, y_pred = _transform_arrays(y_true=y_val, y_pred=y_pred)
        print(
            classification_report(y_val,
                                  y_pred,
                                  target_names=[
                                      self.labels[key]
                                      for key in range(len(self.labels))
                                  ]))
        print(
            confusion_matrix(y_true=y_val,
                             y_pred=y_pred,
                             labels=sorted(self.labels.keys())))
        return

    def get_data(self):
        return self._data


if __name__ == "__main__":
    plot_log('result/log.csv')
