from keras.models import Sequential, Model
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization,
                          Flatten, GlobalMaxPool2D, MaxPool2D, concatenate,
                          Activation, Input, Dense, Dropout, TimeDistributed,
                          Bidirectional, LSTM, GlobalAveragePooling2D, GRU,
                          Convolution1D, MaxPool1D, GlobalMaxPool1D, MaxPooling2D,
                          Reshape, Lambda)
from keras import optimizers
from keras.utils import Sequence, to_categorical
from keras.regularizers import l2
from keras.initializers import random_normal
from keras.activations import relu
from keras import backend as K
from utils import *

# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # hack for load_model
    # import tensorflow as tf

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    # print("CTC lambda inputs / shape")
    # print("y_pred:",y_pred.shape)  # (?, 778, 30)
    # print("labels:",labels.shape)  # (?, 80)
    # print("input_length:",input_length.shape)  # (?, 1)
    # print("label_length:",label_length.shape)  # (?, 1)

    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc(y_true, y_pred):
    return y_pred

def crnn(training=True):
    inp = Input(shape=(None,120,1), name='input')

    x = Convolution2D(16, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(16, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)
    # x = Dropout(0.2)(x)

    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)
    # x = Dropout(0.2)(x)

    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,2))(x)
    # x = Dropout(0.2)(x)

    x = Reshape((-1, 7*128))(x)

    x = TimeDistributed(Dense(256, activation='relu'))(x)
    x = LSTM(256, return_sequences=True, activation='tanh')(x)
    # x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    # x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    pred = TimeDistributed(Dense(len(alphabet) + 1, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.01)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

# model = crnn()
# print(model.summary())
