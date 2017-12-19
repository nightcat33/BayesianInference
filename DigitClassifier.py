'''
    This file is used for part1.1
'''

import math
from matplotlib import pyplot as plt

class DigitClassifier:

    def __init__(self):

        # P(Fij = f | class) for each class
        # _likelihood[digit][i][j]
        self._likelihoods = [
            [
                [
                # 0 - background
                # 1 - foreground 
                    {
                        '0': 0,
                        '1': 0
                    }for i in range (28)
                ]for i in range(28)
            ]for i in range(10)
        ]

        # used to track # of each number in training set
        self._counts = [0 for i in range(10)]
        self._priors = [0 for i in range(10)]
        # two-dimensional array
        # 10 * 10
        self._confusionMatrix = [
            [0 for i in range(10)]
            for i in range(10)
        ]

        self._testSet = [
            [
                [None for i in range(28)]
                for i in range(28)
            ]
            for i in range(1000)
        ]

        self._y_test = [0 for i in range(1000)]


    def train(self, X_train, y_train, k):
        for label in y_train:
            digit = int(label.strip())

            self._counts[digit] += 1

            for i in range(28):
                line = X_train.readline()
                for j in range(28):
                    f = None
                    if line[j] == ' ':
                        f = '0'
                    else:
                        f = '1'
                    # calculate # of times pixel (i,j) has value f 
                    # in training examples from this class
                    self._likelihoods[digit][i][j][f] += 1

        # now doing laplacian smooth and calculate likelihoods
        for i in range(10):
            for j in range(28):
                for q in range(28):
                    # for diff f
                    self._likelihoods[i][j][q]['0'] = math.log2(self._likelihoods[i][j][q]['0']+k) - math.log2(self._counts[i] + k*2)

                    self._likelihoods[i][j][q]['1'] = math.log2(self._likelihoods[i][j][q]['1']+k) - math.log2(self._counts[i] + k*2)

        # calculate priors
        total = sum(self._counts)
        for i in range(10):
            self._priors[i] = self._counts[i] / total

    def loadTest(self, X_test, y_test):
        idx = 0
        for label in y_test:
            self._y_test[idx] = int(label.strip())

            for j in range(28):
                line = X_test.readline()
                for k in range(28):
                    f = None
                    if line[k] == ' ':
                        f = '0'
                    else:
                        f = '1'

                    self._testSet[idx][j][k] = f
            idx += 1

        # print(self._testSet)


    def evaluation(self):
        correct = 0 
        highest_posterior_probs = [[-math.inf, None] for i in range(10)]
        lowest_posterior_probs = [[math.inf, None] for i in range(10)]
        idx = 0
        correctForEachDigits = [0 for i in range(10)]
        totalForEachDigits = [0 for i in range(10)]

        for label in self._y_test:

            predict = None
            all_probs = []
            probs = []
            # assign the image to digit with highest posterior
            for i in range(10):
                prior = math.log2(self._priors[i])
                # calculate logP(wi | class)
                for j in range(28):
                    for k in range(28):
                        f = self._testSet[idx][j][k]
                        prior += self._likelihoods[i][j][k][f]

                prob_map = {
                    'prob': prior,
                    'digit': i
                }
                all_probs.append(prob_map)
                probs.append(prior)

            max_prob = None
            min_prob = None

            max_prob = max(probs)
            min_prob = min(probs)

            for each in all_probs:
                if each['prob'] == max_prob:
                    predict = each['digit']
                    break

            if max_prob > highest_posterior_probs[label][0]:
                highest_posterior_probs[label][0] = max_prob
                highest_posterior_probs[label][1] = idx

            if min_prob < lowest_posterior_probs[label][0]:
                lowest_posterior_probs[label][0] = min_prob
                lowest_posterior_probs[label][1] = idx

            self._confusionMatrix[label][predict] += 1
            
            if predict == label:
                correct += 1
                correctForEachDigits[label] += 1
            totalForEachDigits[label] += 1
            idx += 1

        correct_rate = correct / idx

        print('Correct percentage for test images: ', correct_rate)
        print('Classification rate for each digit:')
        for i in range(10):
            print(i, correctForEachDigits[i]/totalForEachDigits[i])

        print('Highest posterior probabilities.')
        print(highest_posterior_probs)

        print('Lowest posterior probabilities.')
        print(lowest_posterior_probs)

        # now calculate confusion rate
        #  entry in row r and column c is the percentage of
        # test images from class r that are classified as class c
        print("Confusion Matrix:")
        print(self._confusionMatrix)
        for i in range(10):
            for j in range(10):
                self._confusionMatrix[i][j] /= idx
        
        for i in range(10):
            print(self._confusionMatrix[i])   

        confusionMatrixSort = []
        for i in range(10):
            for j in range(10):
                if i != j:
                    confusionMatrixSort.append(((i,j), self._confusionMatrix[i][j]))
        # sort 
        confusionMatrixSort.sort(key = lambda x: -x[1])

        # print(confusionMatrixSort)
        # now we can get four pairs of digits that have the highest confusion rates
        digits_pairs = []
        for i in range(4):
            digits_pairs.append(confusionMatrixSort[i])

        print(digits_pairs)

        for i in range(4):
            feature_a = []
            feature_b = []
            diff = []
            for j in range(28):
                a = []
                b = []
                c = []
                for k in range(28):
                    a.append(self._likelihoods[digits_pairs[i][0][0] ][j][k]['1'])
                    b.append(self._likelihoods[digits_pairs[i][0][1] ][j][k]['1'])
                    c.append(a[-1]-b[-1])
                feature_a.append(a)
                feature_b.append(b)
                diff.append(c)

            figs = [None for q in range(3)]
            axs = [None for q in range(3)]
            colormaps = [None for q in range(3)]
            # likelihoods for two class and odd ratio

            features = [feature_a, feature_b, diff]

            for q in range(3):
                figs[q], axs[q] = plt.subplots()
                colormaps[q] = axs[q].pcolor(features[q], cmap='gist_ncar')
                plt.colorbar(colormaps[q])
                axs[q].invert_yaxis()
                plt.show()
                

if __name__ == '__main__':

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')

    classifier = DigitClassifier()
    classifier.train(X_train, y_train, 0.1)
    classifier.loadTest(X_test, y_test)
    classifier.evaluation()



