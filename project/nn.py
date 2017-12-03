from PIL import Image
import scipy
import glob
import os, os.path
import argparse
import pickle
import gzip
import json
from collections import Counter, defaultdict
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import AvgPool2D
from PIL import Image
import random



class test:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class CNN:
    '''
    CNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, epoches = 5, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epoches = epoches
        self.train_x = train_x
        self.test_x  = test_x
        self.train_y = train_y
        self.test_y = test_y

        # build  CNN model

        self.model = Sequential()

        self.model.add(Conv2D(20, (5, 5),input_shape=(150,150,3)))
        self.model.add(Conv2D(20, (5, 5)))
        self.model.add(Conv2D(20, (5, 5)))
        self.model.add(Conv2D(20, (5, 5)))

        #self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        # self.model.add(Dense(64))
        self.model.add(Dense(9))
        self.model.add(Dense(2))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def train(self):

        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''
        # TODO: fit in training data
        self.model.fit(self.train_x, self.train_y, self.batch_size, self.epoches)

        pass

    def evaluate(self):

        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        predicted = self.model.predict(self.test_x)
        j = 0
        for i in predicted:
            print(self.test_y[j])
            print('predicted')
            print(i)
            j = j+1

        acc = self.model.evaluate(self.test_x, self.test_y,verbose=10)
        return acc

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='CNN classifier options')
    # parser.add_argument('--limit', type=int, default=-1,
    #                     help='Restrict training to this many examples')
    # args = parser.parse_args()

    train_x = []
    train_y = []
    path = "/Users/koushikreddy/Downloads/image"

    list = [i[:-4] for i in os.listdir(path)]
    list.sort()

    for f in list:
        val = []
        for j in f.split(","):
            val.append(j)

        train_y.append(val[0])
        train_y.append(val[2])
        f = str(f)+'.jpg'
        image =  scipy.misc.imread(os.path.join(path, f))
        train_x.append(image)

    random.shuffle(train_x)

    train_x = np.array(train_x)
    print(train_x.shape)
    print(train_x[0][0][0][0])
    train_x = train_x/255;
    print(train_x[0][0][0][0])
    print(len(train_x))
    train_y = np.array(train_y).reshape(len(train_x),2)
    print(train_y.shape)

    cnn = CNN(train_x[:5000], train_y[:5000], train_x[:5001], train_y[:5001])
    cnn.train()
    acc = cnn.evaluate()
    print(acc)
