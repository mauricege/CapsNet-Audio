# CapsNet Audio
This is a fork of the [CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras.git) repository by [XifengGuo](https://github.com/XifengGuo) that adds support for training CapsNet on audio data. It includes scripts for spectrogram extraction and oversampling. This code was used for the paper:
> Shahin Amiriparian, Arsany Awad, Maurice Gerczuk, Lukas Stappen, Alice Baird, Sandra Ottl, Björn Schuller. Audio-based Recognition of Bipolar Disorder Utilising Capsule Networks. In Proceedings of IJCNN


## Usage

**Step 1: Install requirements**

For running the Tool, the following python packages are required:
- tensorflow
- scikit-learn
- imbalanced-learn
- tqdm
- librosa
- pandas


**Step 2: Clone repository**
```
git clone https://github.com/mauricege/CapsNet-Audio.git
cd CapsNet-Audio
```

**Step 3: Spectrogram Extraction**
You need csv files specifying the label for each of your audio input files in a specific partition. E.g., if your training audio files are structured in class folders `data/audio/training/{classname}/*.wav` your csv file (`train.csv` in the following) should look like this:
```
dog/1.wav,dog
cat/2.wav,cat
```
You would then generate spectrograms for the dataset with the command:
```bash
python preprocess/gen_spectrogram.py --window-size 2 --hop-length 1 --sample-rate 48000 --data-setup-file train.csv --out-data data.train.npy --base-dir data/audio/training --out-labels labels.training.npy
```
You can adapt the names of the output files, window size, hop length and sample rate to your liking. Afterwards, extract spectrograms for your validation partition(s). For better training results, you should upsample the training data if you have an imbalanced class distribution in your dataset:
```bash
python preprocess/oversample.py --dataset data.train.npy --labels labels.train.npy --out-data data.train.oversampled.py --out-labels labels.train.oversampled.py
```

**Step 4: Train a CapsNet on the generated spectrograms**  

Training with default settings:
```
python capsulenet.py --data-train data.train.oversampled.py.npy --labels-train labels.train.oversampled.npy --data-test data.dev.npy --labels-test labels.dev.npy
```

More detailed usage run for help:
```
python capsulenet.py --help
```





