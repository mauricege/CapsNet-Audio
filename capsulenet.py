#!python
"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`

This is a forked version for training on audio spectrograms. 
"""

import numpy as np
import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras import callbacks
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical, normalize, multi_gpu_model
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from utils import plot_log, MetricCallback
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256,
                          kernel_size=9,
                          strides=1,
                          padding='valid',
                          activation='relu',
                          name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1,
                             dim_capsule=8,
                             n_channels=32,
                             kernel_size=9,
                             strides=2,
                             padding='valid')
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class,
                             dim_capsule=16,
                             routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class, ))
    masked_by_y = Mask()(
        [digitcaps, y]
    )  # The true label is used to mask the output of capsule layer. For training
    masked = Mask(
    )(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def train(model, eval_model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test), classes = data

    print("x_train {}, y_train {}, x_test {}, y_test {}".format(
        x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size,
                               histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir +
                                           '/weights-{epoch:02d}.h5',
                                           monitor='val_rec_macro',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.lr * (args.lr_decay**epoch))

    terminate_on_nan = callbacks.TerminateOnNaN()

    if os.path.isfile(args.save_dir + '/trained_model.h5'):
        model.load_weights(args.save_dir + '/trained_model.h5')
    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})
    mc = MetricCallback(validation_data=((x_test, y_test), (y_test, x_test)),
                        labels=classes,
                        batch_size=args.batch_size)
    model.fit([x_train, y_train], [y_train, x_train],
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]],
              callbacks=[mc, log, tb, checkpoint, lr_decay],
              shuffle=True)

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/log.csv', show=True)

    y_pred = eval_model.predict(
        x_test, batch_size=args.batch_size)[0].astype("float32")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Recall score: {:.2f}%".format(recall * 100))
    print("Confusion matrix:\n{}".format(cm))

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Recall score: {:.2f}%".format(recall * 100))
    print("Confusion matrix:\n{}".format(cm))


def load_audiodata(args):
    #1 load training data
    x_train = np.load(args.data_train)

    #x_train = (x_train - min_train) / (max_train - min_train)
    y_train = np.load(args.labels_train)
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2],
                              1).astype("float32")

    #2 load test data
    x_test = np.load(args.data_test)
    #x_test = (x_test - min_train) / (max_train - min_train)
    y_test = np.load(args.labels_test)
    y_test = lb.transform(y_test)
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2],
                            1).astype("float32")

    print("Training dataset {}x{}x{}x{} .. labels {}".format(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3],
        y_train.shape))
    print("Test dataset {}x{}x{}x{} .. labels {}".format(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3],
        y_test.shape))
    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(x_test))
    return x_train, y_train, x_test, y_test, lb.classes_


if __name__ == "__main__":

    # setting the hyper parameters
    parser = argparse.ArgumentParser(
        description="Capsule Network on 3D Audio data.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help="Initial learning rate")
    parser.add_argument(
        '--lr-decay',
        default=0.9,
        type=float,
        help=
        "The value multiplied by lr at each epoch. Set a larger value for larger epochs"
    )
    parser.add_argument('--lam-recon',
                        default=0.392,
                        type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument(
        '-r',
        '--routings',
        default=3,
        type=int,
        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug',
                        action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t',
                        '--testing',
                        action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument(
        '-w',
        '--weights',
        default=None,
        help="The path of the saved weights. Should be specified when testing")

    parser.add_argument('-tr',
                        '--data-train',
                        default=None,
                        help="Training dataset numpy file")
    parser.add_argument('-l-tr',
                        '--labels-train',
                        default=None,
                        help="Training labels numpy file")
    parser.add_argument('-te',
                        '--data-test',
                        default=None,
                        help="Test dataset numpy file")
    parser.add_argument('-l-te',
                        '--labels-test',
                        default=None,
                        help="Test labels numpy file")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data and define model
    x_train, y_train, x_test, y_test, classes = load_audiodata(args)

    model, eval_model, manipulate_model = CapsNet(
        input_shape=x_train.shape[1:],
        n_class=int(y_train.shape[1]),
        routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model,
              eval_model=eval_model,
              data=((x_train, y_train), (x_test, y_test), classes),
              args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print(
                'No weights are provided. Will test using random initialized weights.'
            )
        test(model=eval_model, data=(x_test, y_test), args=args)
