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

X = ['zxc', 'xcb', 'zxcb', 'z', 'bc']

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
import json
def make_y_from_json(json_file, out):
    with open(json_file, encoding='utf-8') as f:
        y = json.load(f).values()
        y = label_encoder(y)

        joblib.dump(y, out)

import os
import cv2
from helpers import resize
def preprocess(img_dir):
    img = cv2.imread(img_dir, 1)
    img = resize(img, 120, always=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 71, 17)

    return img.swapaxes(0,1)[:,:,None] / 255 * 2 - 1

def make_X_from_images(dir, out):
    f = open(os.path.join(dir, 'labels.json'), encoding='utf-8')
    files = json.load(f).keys()
    f.close()

    X = []
    for img in files:
        X.append(preprocess(os.path.join(dir, img)))

    joblib.dump(X, out)

# make_y_from_json('0916_Data Samples 2/labels.json', 'y_2')
# y = joblib.load('y_2')
# print(type(y[0]))

from sklearn.model_selection import StratifiedKFold, train_test_split
class DataGenerator:

    def __init__(self, X_file, y_file, desample_factor, val_size=0.2):
        X = joblib.load(X_file)
        y = joblib.load(y_file)

        self.X_train, self.X_val, self.y_train, self.y_val = \
        train_test_split(X, y, test_size=val_size, shuffle=True)

        self.desample_factor = desample_factor
        self.train_size = len(self.X_train)
        self.val_size = len(self.X_val)

    def next_train(self):
        while True:
            for i in range(self.train_size):
                X = np.array(self.X_train[i:i+1])
                y = np.array(self.y_train[i:i+1])

                input_length = np.ones([1, 1]) * (X.shape[1] // self.desample_factor - 2)
                label_length = np.ones([1, 1]) * len(y[0])

                inputs = {
                    'input': X,
                    'labels': y,
                    'input_length': input_length,
                    'label_length': label_length,
                    }

                outputs = {'ctc': np.zeros([1])}

                yield (inputs, outputs)

    def next_val(self):
        while True:
            for i in range(self.val_size):
                X = np.array(self.X_val[i:i+1])
                y = np.array(self.y_val[i:i+1])

                input_length = np.ones([1, 1]) * (X.shape[1] // self.desample_factor - 2)
                label_length = np.ones([1, 1]) * len(y[0])

                inputs = {
                    'input': X,
                    'labels': y,
                    'input_length': input_length,
                    'label_length': label_length,
                    }

                outputs = {'ctc': np.zeros([1])}

                yield (inputs, outputs)
