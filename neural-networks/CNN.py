
import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.core import Reshape

class Numbers:
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
    def __init__(self, train_x, train_y, test_x, test_y, epoches = 15, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epoches = epoches
        # TODO: reshape train_x and test_x
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length
        # normalize data to range [0, 1]

        self.train_x = train_x.reshape(len(train_x), 28, 28, 1)
        self.test_x = test_x.reshape(len(test_x),28,28,1)

        # TODO: one hot encoding for train_y and test_y

        length = len(train_y)
        self.train_y = [[0 for i in range(10)] for j in range(length)]

        for i in range(0,length,1):
            self.train_y[i][train_y[i]] = 1

        length = len(test_y)
        self.test_y = [[0 for i in range(10)] for j in range(length)]

        for i in range(0, length, 1):
            self.test_y[i][test_y[i]] = 1

        # TODO: build you CNN model

        self.model = Sequential()
        self.model.add(Conv2D(20, (4, 4) ,input_shape=(28, 28, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(20, (4, 4)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        # self.model.add(Dense(64))
        # self.model.add(Activation('relu'))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #self.model = model
        # self.train_x /= 255
        # self.test_x /= 255

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
        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")


    cnn = CNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    cnn.train()
    acc = cnn.evaluate()
    print(acc)
