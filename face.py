
'''
    This file is for extra credit of part1
'''

import math
from matplotlib import pyplot as plt
import numpy as np

class DigitClassifier:

    def __init__(self):

        # P(Fij = f | class) for each class
        # _likelihood[digit][i][j]
        self._likelihoods = [
            [
                [
                    {
                        '#': 0,
                        ' ': 0
                    }for i in range (60)
                ]for i in range(70)
            ]for i in range(2)
        ]

        # used to track # of each number in training set
        self._counts = [0 for i in range(2)]
        self._priors = [0 for i in range(2)]

        self._testSet = [
            [
                [None for i in range(60)]
                for i in range(70)
            ]
            for i in range(150)
        ]

        self._y_test = [0 for i in range(150)]


    def train(self, X_train, y_train, k):
        for label in y_train:
            digit = int(label.strip())

            self._counts[digit] += 1

            for i in range(70):
                line = X_train.readline()
                for j in range(60):
                    self._likelihoods[digit][i][j][line[j]] += 1

        # now doing laplacian smooth and calculate likelihoods
        for i in range(2):
            for j in range(70):
                for q in range(60):
                    # for diff f
                    self._likelihoods[i][j][q]['#'] = math.log2(self._likelihoods[i][j][q]['#']+k) - math.log2(self._counts[i] + k*2)

                    self._likelihoods[i][j][q][' '] = math.log2(self._likelihoods[i][j][q][' ']+k) - math.log2(self._counts[i] + k*2)

        # calculate priors
        total = sum(self._counts)
        for i in range(2):
            self._priors[i] = self._counts[i] / total

    def loadTest(self, X_test, y_test):
        idx = 0
        for label in y_test:
            self._y_test[idx] = int(label.strip())

            for j in range(70):
                line = X_test.readline()
                for k in range(60):
                    self._testSet[idx][j][k] = line[k]
            idx += 1

        # print(self._testSet)
    def evaluation(self):
        correct = 0 
        idx = 0

        for label in self._y_test:

            predict = None
            max_prob = -math.inf
            # assign the image to digit with highest posterior
            for i in range(2):
                prior = math.log2(self._priors[i])
                # calculate logP(wi | class)
                for j in range(70):
                    for k in range(60):
                        f = self._testSet[idx][j][k]
                        prior += self._likelihoods[i][j][k][f]


                if prior > max_prob:
                    max_prob = prior
                    predict = i
            
            if predict == label:
                correct += 1
            idx += 1

        correct_rate = correct / idx

        print('Correct percentage for test images: ', correct_rate)


if __name__ == '__main__':

    X_train = open('./facedata/facedatatrain', 'r') 
    y_train = open('./facedata/facedatatrainlabels', 'r')
    X_test = open('./facedata/facedatatest', 'r')
    y_test = open('./facedata/facedatatestlabels', 'r')

    classifier = DigitClassifier()
    classifier.train(X_train, y_train, 3)
    classifier.loadTest(X_test, y_test)
    classifier.evaluation()
