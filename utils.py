import numpy as np

# 207 characters
alphabet = [' ', ',', '.', '/', '+', '-', '#', "'", '(', ')', ':',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư',
            'V', 'W', 'X', 'Y', 'Z',
            'À', 'Á', 'Ả', 'Ã', 'Ạ',
            'Ằ', 'Ắ', 'Ẳ', 'Ẵ', 'Ặ',
            'Ầ', 'Ấ', 'Ẩ', 'Ẫ', 'Ậ',
            'È', 'É', 'Ẻ', 'Ẽ', 'Ẹ',
            'Ề', 'Ế', 'Ể', 'Ễ', 'Ệ',
            'Ì', 'Í', 'Ỉ', 'Ĩ', 'Ị',
            'Ò', 'Ó', 'Ỏ', 'Õ', 'Ọ',
            'Ồ', 'Ố', 'Ổ', 'Ỗ', 'Ộ',
            'Ờ', 'Ớ', 'Ở', 'Ỡ', 'Ợ',
            'Ù', 'Ú', 'Ủ', 'Ũ', 'Ụ',
            'Ừ', 'Ứ', 'Ử', 'Ữ', 'Ự',
            'Ỳ', 'Ý', 'Ỷ', 'Ỹ', 'Ỵ',
            'a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'ô', 'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư',
            'v', 'w', 'x', 'y', 'z',
            'à', 'á', 'ả', 'ã', 'ạ',
            'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
            'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ',
            'è', 'é', 'ẻ', 'ẽ', 'ẹ',
            'ề', 'ế', 'ể', 'ễ', 'ệ',
            'ì', 'í', 'ỉ', 'ĩ', 'ị',
            'ò', 'ó', 'ỏ', 'õ', 'ọ',
            'ồ', 'ố', 'ổ', 'ỗ', 'ộ',
            'ờ', 'ớ', 'ở', 'ỡ', 'ợ',
            'ù', 'ú', 'ủ', 'ũ', 'ụ',
            'ừ', 'ứ', 'ử', 'ữ', 'ự',
            'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ',]
idx = {alphabet[i]: i for i in range(len(alphabet))}
blank_idx = len(alphabet)

desample_factor = {
    'model0': 3*2*2*2,
    'model1': 3*2*2*2,
    'model2': 3*2*2*2,
    'model3': 3*2*2*2,
    'model4': 2*2*2*2,
    'model5': 2*2*2*1,
    'crnn': 2*2*2*1,
}

def string_vectorizer(str, alphabet):
    labels = [idx[letter] for letter in str]
    return np.array(labels)

def label_encoder(y, alphabet=alphabet):
    return [string_vectorizer(s, alphabet) for s in y]

import itertools
def ctc_decoder(pred):
    out_best = list(np.argmax(pred[0, 2:], axis=1))
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    text = ''
    for i in out_best:
        if i < len(alphabet):
            text += alphabet[i]
    return text

from sklearn.externals import joblib
from keras.preprocessing.sequence import pad_sequences
import json
text_len = 69
def make_y_from_json(json_file, out):
    with open(json_file, encoding='utf-8') as f:
        y = json.load(f).values()
        y = label_encoder(y)
        y = pad_sequences(y, padding='post', value=blank_idx)
        joblib.dump(y, out)

import os
import cv2
from helpers import resize
img_w = 2203
img_h = 120
def preprocess(img_dir, padding=True):
    img = cv2.imread(img_dir, 1)
    img = resize(img, img_h, always=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 71, 17)
    if padding:
        img = cv2.copyMakeBorder(img, 0, 0, 0, img_w - img.shape[1], cv2.BORDER_CONSTANT, value=0)

    return img.swapaxes(0,1)[:,:,None] / 255 * 2 - 1

def make_X_from_images(dir, out):
    f = open(os.path.join(dir, 'labels.json'), encoding='utf-8')
    files = json.load(f).keys()
    f.close()

    X = np.zeros((len(files), img_w, img_h))
    for i in range(len(files)):
        X[i] = preprocess(os.path.join(dir, files[i]))
    X = np.array(X)

    np.save(out, X)

# make_y_from_json('0916_Data Samples 2/labels.json', 'y_2')
# y = joblib.load('y_2')
# print(type(y[0]))

from sklearn.model_selection import StratifiedKFold, train_test_split
import math
class DataGenerator:

    def __init__(self, X_file, y_file, desample_factor, batch_size=32, val_size=0.2):
        X = np.load(X_file + '.npy')
        y = np.load(y_file + '.npy')

        self.input_length = X.shape[1] // desample_factor - 2

        self.X_train, self.X_val, self.y_train, self.y_val = \
        train_test_split(X, y, test_size=val_size, shuffle=True)

        # self.desample_factor = desample_factor
        self.batch_size = batch_size
        self.train_size = len(self.X_train)
        self.val_size = len(self.X_val)
        self.train_steps = math.ceil(self.train_size / batch_size)
        self.val_steps = math.ceil(self.val_size / batch_size)

    def next_train(self):
        while True:
            for i in range(0, self.train_size, self.batch_size):
                X = self.X_train[i : i+self.batch_size]
                y = self.y_train[i : i+self.batch_size]

                batch_size = len(X)

                input_length = np.ones([batch_size, 1]) * self.input_length
                label_length = np.array([sum(label != -1) for label in y])[:None]

                inputs = {
                    'input': X,
                    'labels': y,
                    'input_length': input_length,
                    'label_length': label_length,
                    }

                outputs = {'ctc': np.zeros([batch_size])}

                yield (inputs, outputs)

    def next_val(self):
        while True:
            for i in range(0, self.val_size, self.batch_size):
                X = self.X_val[i : i+self.batch_size]
                y = self.y_val[i : i+self.batch_size]

                batch_size = len(X)

                input_length = np.ones([batch_size, 1]) * self.input_length
                label_length = np.array([sum(label != -1) for label in y])[:None]

                inputs = {
                    'input': X,
                    'labels': y,
                    'input_length': input_length,
                    'label_length': label_length,
                    }

                outputs = {'ctc': np.zeros([batch_size])}

                yield (inputs, outputs)
