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

    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc(y_true, y_pred):
    return y_pred

def model0(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,2))(x)

    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Reshape((-1, 7*128))(x)

    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
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

    desample_factor = 3*2*2*2

def model1(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,2))(x)

    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    # x = Dropout(0.2)(x)

    x = Reshape((-1, 7*128))(x)

    x = LSTM(256, return_sequences=True, activation='tanh')(x)
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

def model2(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(8, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,2))(x)

    x = Convolution2D(16, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(32, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Reshape((-1, 7*64))(x)

    x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
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

def model3(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (5,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,2))(x)

    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

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
    x = Dropout(0.2)(x)

    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

    x = Reshape((-1, 7*128))(x)

    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='sum')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    x = Dropout(0.2)(x)
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
        opt = optimizers.Nadam(0.001)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def model4(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (10,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(32, (10,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (10,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(64, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(128, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

    x = Reshape((-1, 7*128))(x)

    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='sum')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    x = Dropout(0.2)(x)
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
        opt = optimizers.Nadam(0.001)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def crnn(training=True):
    inp = Input(shape=(None,img_h,1), name='input')

    x = Convolution2D(16, (9,19), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(16, (9,19), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(32, (9,19), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (9,19), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(64, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(128, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(128, (5,11), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,1))(x)
    x = Dropout(0.5)(x)

    x = Reshape((-1, 15*128))(x)

    x = TimeDistributed(Dense(512, activation='elu'))(x)
    # x = Dropout(0.2)(x)
    # x = LSTM(256, return_sequences=True, activation='tanh')(x)
    # x = LSTM(128, return_sequences=True, activation='tanh')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    # x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    x = Dropout(0.5)(x)
    # x = TimeDistributed(Dense(256, activation='relu'))(x)
    # x = Dropout(0.2)(x)
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
        opt = optimizers.Nadam(0.001)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model

models = {
    'model0': model0,
    'model1': model1,
    'model2': model2,
    'model3': model3,
    'model4': model4,
    'crnn': crnn,
}

# model = crnn()
# print(model.summary())
