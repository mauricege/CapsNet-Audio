from imblearn.over_sampling import RandomOverSampler
import numpy as np
import argparse


def oversample(data, labels):
    data_idx = np.reshape(range(data.shape[0]), (-1, 1))
    ros = RandomOverSampler(ratio="auto")
    print("Oversampling {} samples: {}".format(
        labels.shape, np.unique(labels, return_counts=True)))
    data_idx_o, labels_o = ros.fit_sample(data_idx, labels)
    print("Result {} samples: {}".format(
        labels_o.shape, np.unique(labels_o, return_counts=True)))
    return data[data_idx_o.flatten()], labels_o


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Example usage: python oversample.py -d training.npy -l labels_training.npy -od training.oversampled.py -ol labels_training.oversampled.py"
    )
    parser.add_argument("-d", "--dataset", type=str, help="Dataset .npy file")
    parser.add_argument("-l", "--labels", type=str, help="Labels .npy file")
    parser.add_argument("-od",
                        "--output-dataset",
                        type=str,
                        help="Output dataset filename")
    parser.add_argument("-ol",
                        "--output-labels",
                        type=str,
                        help="Output labels filename")
    args = parser.parse_args()

    data = np.load(args.dataset)
    labels = np.load(args.labels)

    data_o, labels_o = oversample(data, labels)
    np.save(args.output_dataset, data_o)
    np.save(args.output_labels, labels_o)
