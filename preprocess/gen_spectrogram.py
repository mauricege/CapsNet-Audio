import argparse
import os
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer


def generate_dataset(args):
    dataset = []
    labels = []
    fileslist = open(args.data_setup_file, "r").readlines()
    for line in tqdm(fileslist, desc='Extracting spectrograms'):
        wavfile, label = line.rstrip().split(",")
        fullpath = os.path.join(args.base_dir, wavfile)
        y, sr = librosa.load(fullpath, sr=args.sample_rate)
        frames = librosa.util.frame(y,
                                    frame_length=int(args.window_size * sr),
                                    hop_length=int(args.hop_length * sr))
        for frame in frames.T:
            spc = librosa.feature.melspectrogram(y=frame, sr=sr)
            dataset.append(spc)
            labels.append(label)

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Example usage: python gen_spectrogram.py -w 2 -hl 1 -s 48000 -d training.csv -od training.npy -bd Mani/ -ol labels_training.npy"
    )
    parser.add_argument("-w",
                        "--window-size",
                        default=2,
                        type=float,
                        help="Window Size (int)")
    parser.add_argument("-hl",
                        "--hop-length",
                        default=1,
                        type=int,
                        help="Hop Length (int)")
    parser.add_argument("-s",
                        "--sample-rate",
                        default=48000,
                        type=int,
                        help="Sample Rate (int)")
    parser.add_argument("-d",
                        "--data-setup-file",
                        type=str,
                        help="CSV file for the setup of the dataset")
    parser.add_argument(
        "-bd",
        "--base-dir",
        type=str,
        help="Base Directory of the audio files of the dataset")
    parser.add_argument("-od",
                        "--out-data",
                        type=str,
                        help="Filename for output dataset (.npy)")
    parser.add_argument("-ol",
                        "--out-labels",
                        type=str,
                        help="Filename for output labels (.npy)")
    args = parser.parse_args()

    dataset, labels = generate_dataset(args)
    print("final dataset: {}".format(dataset.shape))
    print("final labels: {}".format(labels.shape))
    np.save(args.out_data, dataset)
    np.save(args.out_labels, labels)
