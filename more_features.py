'''
    This file is for extra credit of part1
'''
import math

class DigitClassifier:

    def __init__(self):

        # P(Fij = f | class) for each class
        # _likelihood[digit][i][j]
        self._likelihoods = [
            [
                [
                    {
                        '#': 0,
                        '+': 0,
                        ' ': 0
                    }for i in range (28)
                ]for i in range(28)
            ]for i in range(10)
        ]

        # used to track # of each number in training set
        self._counts = [0 for i in range(10)]
        self._priors = [0 for i in range(10)]

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
                    # calculate # of times pixel (i,j) has value f 
                    # in training examples from this class
                    self._likelihoods[digit][i][j][line[j]] += 1

        # now doing laplacian smooth and calculate likelihoods
        for i in range(10):
            for j in range(28):
                for q in range(28):
                    # for diff f
                    self._likelihoods[i][j][q]['#'] = math.log2(self._likelihoods[i][j][q]['#']+k) - math.log2(self._counts[i] + k*3)
                    self._likelihoods[i][j][q]['+'] = math.log2(self._likelihoods[i][j][q]['+']+k) - math.log2(self._counts[i] + k*3)
                    self._likelihoods[i][j][q][' '] = math.log2(self._likelihoods[i][j][q][' ']+k) - math.log2(self._counts[i] + k*3)

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
                    self._testSet[idx][j][k] = line[k]
            idx += 1

    def evaluation(self):
        correct = 0 

        idx = 0

        for label in self._y_test:

            predict = None
            max_prob = -math.inf
            # assign the image to digit with highest posterior
            for i in range(10):
                prior = math.log2(self._priors[i])
                # calculate logP(wi | class)
                for j in range(28):
                    for k in range(28):
                        f = self._testSet[idx][j][k]
                        prior += self._likelihoods[i][j][k][f]

                if prior > max_prob:
                    predict = i
                    max_prob = prior
            
            if predict == label:
                correct += 1

            idx += 1

        correct_rate = correct / idx

        print('Correct percentage for test images: ', correct_rate)


if __name__ == '__main__':

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')

    classifier = DigitClassifier()
    classifier.train(X_train, y_train, 0.1)
    classifier.loadTest(X_test, y_test)
    classifier.evaluation()


