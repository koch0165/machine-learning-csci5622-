import random
import argparse
import gzip
import pickle

import numpy as np
from math import exp, log
from collections import defaultdict


SEED = 1735

random.seed(SEED)


class Numbers:
    """
    Class to store MNIST data for images of 0 and 1 only
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if you'd like

        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        train_indices = np.where(self.train_y > 7)
        self.train_x, self.train_y = self.train_x[train_indices], self.train_y[train_indices]
        self.train_y = self.train_y - 8

        self.valid_x, self.valid_y = valid_set
        valid_indices = np.where(self.valid_y > 7)
        self.valid_x, self.valid_y = self.valid_x[valid_indices], self.valid_y[valid_indices]
        self.valid_y = self.valid_y - 8

        self.test_x, self.test_y = test_set
        test_indices = np.where(self.test_y > 7)
        self.test_x, self.test_y = self.test_x[test_indices], self.test_y[test_indices]
        self.test_y = self.test_y - 8

    @staticmethod
    def shuffle(X, y):
        """ Shuffle training data """
        shuffled_indices = np.random.permutation(len(y))
        return X[shuffled_indices], y[shuffled_indices]


class LogReg:
    def __init__(self, num_features, eta):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param eta: A function that takes the iteration as an argument (the default is a constant value)
        """

        self.w = np.zeros(num_features)
        self.eta = eta
        self.last_update = defaultdict(int)

    def progress(self, examples_x, examples_y):
        """
        Given a set of examples, compute the probability and accuracy
        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for x_i, y in zip(examples_x, examples_y):
            p = sigmoid(self.w.dot(x_i))
            if y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples_y))

    def sgd_update(self, x_i, y):

        sigmoid_val = sigmoid(self.w.dot(x_i))

        count = 0
        for i in x_i:
            self.w[count] = self.w[count] + self.eta*(y-sigmoid_val)*i
            count = count + 1

        """
        Compute a stochastic gradient update to improve the log likelihood.
        :param x_i: The features of the example to take the gradient with respect to
        :param y: The target output of the example to take the gradient with respect to
        :return: Return the new value of the regression coefficients
        """

        # TODO: Finish this function to do a single stochastic gradient descent update
        # and return the updated weight vector

        return self.w

def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.
    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * np.sign(score)

    return exp(score)/(1+exp(score))
    # TODO: Finish this function to return the output of applying the sigmoid
    # function to the input score (Please do not use external libraries)

    #return 1.0

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eta", help="Initial SGD learning rate",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--passes", help="Number of passes through training data",
                           type=int, default=1, required=False)

    args = argparser.parse_args()

    data = Numbers('../data/mnist.pkl.gz')

    # Initialize model
    lr = LogReg(data.train_x.shape[1], args.eta)

    # Iterations
    iteration = 0
    for epoch in range(args.passes):
        data.train_x, data.train_y = Numbers.shuffle(data.train_x, data.train_y)
        i = 0
        for train_x,train_y in zip(data.train_x,data.train_y):
            lr.w = lr.sgd_update(train_x,train_y)
            # if(i%100==0):
            #     (logprob, accuracy) = lr.progress(data.test_x, data.test_y)
            #     print("log prob %f accuracy %f" % (logprob, accuracy))
            # i = i+1
        # TODO: Finish the code to loop over the training data and perform a stochastic
        # gradient descent update on each training example.
        # (logprob, accuracy) = lr.progress(data.test_x, data.test_y)
        # print("log prob %f accuracy %f" % (logprob, accuracy))
        # NOTE: It may be helpful to call upon the 'progress' method in the LogReg
        # class to make sure the algorithm is truly learning properly on both training and test data
