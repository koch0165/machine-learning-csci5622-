import os
import json
from csv import DictReader, DictWriter

import numpy as np
from numpy import array
from collections import Counter, defaultdict
import re


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize

SEED = 5


'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        max = -1
        for ex in examples:
            print("count %d"%(i))
            features[i, 0] = len(ex)
            i += 1

        max = features.max(axis=0)[0]
        min = features.min(axis=0)[0]

        for j in range(0,i):
            features[j] = (features[j] - min)/(max-min)

        return features


class PostiveNegativeWordCounter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        posWords = ['first-rate','insightful','clever','charming','comical','charismatic','enjoyable','uproarious','original',
        'tender','hilarious','absorbing','humorous','enjoy','enjoyable','sensitive','riveting','intriguing','powerful','fascinating',
        'fascinating','pleasant','surprising','dazzling','thought provoking','imaginative','legendary','unpretentious','flawless',
        'entertaining']

        negWords = ['second-rate','violent','moronic','third-rate','flawed','juvenile','avoid','pathetic','boring','distasteful','ordinary',
                    'disgusting','depressing','senseless','static','brutal',	'confused','disappointing','bloody','silly','tired','predictable','stupid',
                     'uninteresting','weak','incredibly', 'tiresome','trite','uneven','cliché', 'ridden','outdated','don\'t','can\'t',
                     'dreadful','bland','not','never']

        features = np.zeros((len(examples), 2))
        i = 0
        max = -1
        for ex in examples:
            reg1 = re.compile(r' .*less ')
            reg2 = re.compile(r'un.* ')
            posWords = [s for s in ex.split() if posWords.__contains__(s) ]
            negWords = [s for s in ex.split() if (negWords.__contains__(s) or reg1.match(s) or reg2.match(s))]
            print("count %d"%(i))
            features[i, 0] = len(posWords)
            features[i, 1] = len(negWords)
            i += 1

        max0 = features.max(axis=0)[0]
        min0 = features.min(axis=0)[0]

        max1 = features.max(axis=0)[1]
        min1 = features.min(axis=0)[1]

        for j in range(0, i):
            features[j][0] = (features[j][0] - min0) / (max0 - min0)

        for j in range(0, i):
            features[j][1] = (features[j][1] - min0) / (max1 - min1)

        return features

class ConjunctionCounter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        conjunctions = ['because','but','though','unless','until','since','although','whenever','wherever','while','so','that','nevertheless']
        features = np.zeros((len(examples), 1))
        i = 0
        max = -1

        for ex in examples:
            reg = re.compile(r'so.*that')
            numbers = [s for s in ex.split() if (conjunctions.__contains__(s) or reg.match(s))]
            count = len(numbers)
            features[i, 0] = len(ex)
            i += 1

        max = features.max(axis=0)[0]
        min = features.min(axis=0)[0]

        for j in range(0,i):
            features[j] = (features[j] - min)/(max-min)

        return features

class ParanthesisCounter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        max = -1

        for ex in examples:
            count = ex.count(' ( ')
            features[i, 0] = count
            i += 1

        return features

class ExclamationCountTransformer(BaseEstimator,TransformerMixin) :

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = ex.count('!')
            i += 1

        return features

class SentenceCounter(BaseEstimator,TransformerMixin) :

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            numWords = [len(sentence.split()) for sentence in ex.split('.')]
            avg = float(sum(numWords))/len(numWords)
            features[i, 0] = avg
            i += 1

        return features


class InvitedQuotesCounter(BaseEstimator,TransformerMixin) :

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            count = ex.count('\"')/2
            features[i,0] = count
            i += 1

        return features

class NumberCounter(BaseEstimator, TransformerMixin):

        def __init__(self):
            pass

        def fit(self, examples):
            return self

        def transform(self, examples):
            features = np.zeros((len(examples), 1))
            i = 0
            for ex in examples:
                numbers = [int(s) for s in ex.split() if (s.isdigit() and len(s) == 4) ]
                count = len(numbers)
                features[i, 0] = count
                print("Number count %d %d" % (i, count))
                i += 1

            return features


class AdverbCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        pattern = re.compile(r'.*ly')
        for ex in examples:
            numbers = [s for s in ex.split() if pattern.match(s) ]
            count = len(numbers)
            features[i, 0] = count
            print("Adverb count %d %d" % (i, count))
            i += 1

        return features

class WordLengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            numbers = [s for s in ex.split() if len(s)>=13]
            count = len(numbers)
            features[i, 0] = count
            print("Number count %d %d" % (i, count))
            i += 1

        return features

class LastSentenceCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            sentences = ex.split('.')
            print("Last sentence %s "%(sentences[len(sentences)-2]))
            lastsentence = sentences[len(sentences)-2]
            count = len(lastsentence.split())
            features[i, 0] = count
            print("Number count %d %d" % (i, count))
            i += 1

        max = features.max(axis=0)[0]
        min = features.min(axis=0)[0]

        for j in range(0, i):
            features[j] = (features[j] - min) / (max - min)

        return features

class QuestionCounter(BaseEstimator, TransformerMixin):

            def __init__(self):
                pass

            def fit(self, examples):
                return self

            def transform(self, examples):
                features = np.zeros((len(examples), 1))
                i = 0
                for ex in examples:
                    regex = re.compile(r'(how|what|why|where|whose|who|whom|which|when).*\?')
                    match = regex.findall(ex)
                    count = len(match)
                    features[i, 0] = count
                    if(i>600):
                        print("Invited count %d %d" % (i, count))
                    i += 1

                return features



# TODO: Add custom feature transformers for the movie review data


class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            ('text_stats', Pipeline([
               ('selector', ItemSelector(key='text')),
               ('text_length', TextLengthTransformer())
            ])),
            ('exclamation',Pipeline([
                ('selector', ItemSelector(key='text')),
                ('exclamation_count', ExclamationCountTransformer())])),
             ('invitedcount', Pipeline([
                 ('selector', ItemSelector(key='text')),
                 ('invited_count', InvitedQuotesCounter())])),
            ('question', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('question_count', QuestionCounter())])),
            ('conjunction_count', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('conjunction', ConjunctionCounter())])),
            ('paranthesis_count', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('paranthesis', ParanthesisCounter())])),
            ('ngram', Pipeline([
            ('selector', ItemSelector(key="text")),
            ('vectorizer', CountVectorizer(token_pattern='[A-Z|a-z|0-9|-|\']+(?=\s+|.||\?|!)',
                                           ngram_range=(1, 2)))
            ])),

        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data
    dataset_x = []
    dataset_y = []
    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()

    labels = []
    for l in y_train:
        if not l in labels:
            labels.append(l)

    # Here we collect the train features
    # The inner dictionary contains certain pieces of the input data that we
    # would like to be able to select with the ItemSelector
    # The text key refers to the plaintext
    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })

    #print(feat_train)
    #print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=15000, shuffle=True, verbose=2)

    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)


    # EXTRA CREDIT: Replace the following code with scikit-learn cross validation
    # and determine the best 'alpha' parameter for regularization in the SGDClassifier
