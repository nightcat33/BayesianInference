import math
import time
'''
    This file is for part1.2 - disjoint patches
'''
class DigitClassifier:

    def __init__(self, n, m):

        # P(Fij = f | class) for each class
        # _likelihood[digit][i][j]
        # calculate # of distinct values thta a feature can take
        self._v = math.pow(2, n*m)
        self._n = n
        self._m = m
        # calculate # of row and col for disjoint patches
        self._disj_row = int(28/n)
        self._disj_col = int(28/m)

        self._overlap_row = 28-n+1
        self._overlap_col = 28-m+1

        self._likelihoods = [
            [
                [
                    {

                    }for i in range (self._disj_col)
                ]for i in range(self._disj_row)
            ]for i in range(10)
        ]

        self._likelihoods_overlap = [
            [
                [
                    {

                    }for i in range (self._overlap_col)
                ]for i in range(self._overlap_row)
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

    # this is used for disjoint patches
    def train(self, X_train, y_train, k):
        self._k = k
        t_start = time.time()
        for label in y_train:
            digit = int(label.strip())

            self._counts[digit] += 1

            for i in range(self._disj_row):
                lines = []
                for each in range(self._n):
                    line = X_train.readline()
                    lines.append(line)

                for j in range(self._disj_col):
                    values = []
                    for p in range(self._n):
                        for q in range(self._m):
                            values.append(lines[p][j*self._m + q])
                    feature = tuple(values)
                    # print(feature)
                    if self._likelihoods[digit][i][j].get(feature) == None:
                        self._likelihoods[digit][i][j][feature] = 1
                    else:
                        self._likelihoods[digit][i][j][feature] += 1

        # print(self._likelihoods)

        # now doing laplacian smooth and calculate likelihoods
        for i in range(10):
            for j in range(self._disj_row):
                for q in range(self._disj_col):
                    # for diff f
                    for each in self._likelihoods[i][j][q]:

                        self._likelihoods[i][j][q][each] = math.log2(self._likelihoods[i][j][q][each]+k) - math.log2(self._counts[i] + k*self._v)

        # print(self._likelihoods)
        # calculate priors
        total = sum(self._counts)
        for i in range(10):
            self._priors[i] = self._counts[i] / total
        # print(self._priors)
        t_end = time.time()
        training_time = t_end - t_start
        print("Training time is: %fs" % training_time)



    def train_overlap(self, X_train, y_train, k):
        self._k = k
        t_start = time.time()
        for label in y_train:
            digit = int(label.strip())

            self._counts[digit] += 1
            # read each digit image
            lines = []
            for i in range(28):
                line = X_train.readline()
                lines.append(line)

            for i in range(self._overlap_row):
                
                for j in range(self._overlap_col):
                    values = []
                    for p in range(self._n):
                        for q in range(self._m):
                            values.append(lines[i+p][j+q])

                    feature = tuple(values)
                    # print(feature)
                    if self._likelihoods_overlap[digit][i][j].get(feature) == None:
                        self._likelihoods_overlap[digit][i][j][feature] = 1
                    else:
                        self._likelihoods_overlap[digit][i][j][feature] += 1

        # print(self._likelihoods_overlap)

        # now doing laplacian smooth and calculate likelihoods
        for i in range(10):
            for j in range(self._overlap_row):
                for q in range(self._overlap_col):
                    # for diff f
                    for each in self._likelihoods_overlap[i][j][q]:

                        self._likelihoods_overlap[i][j][q][each] = math.log2(self._likelihoods_overlap[i][j][q][each]+k) - math.log2(self._counts[i] + k*self._v)

        # print(self._likelihoods_overlap)
        # calculate priors
        total = sum(self._counts)
        for i in range(10):
            self._priors[i] = self._counts[i] / total
        # print(self._priors)
        t_end = time.time()
        training_time = t_end - t_start
        print("Training time is: %fs" % training_time)

    def loadTest_overlap(self, X_test, y_test):
        idx = 0
        for label in y_test:
            self._y_test[idx] = int(label.strip())

            lines = []
            for i in range(28):
                line = X_test.readline()
                lines.append(line)

            for i in range(self._overlap_row):

                for j in range(self._overlap_col):
                    values = []
                    for p in range(self._n):
                        for q in range(self._m):
                            values.append(lines[i+p][j+q])

                    feature = tuple(values)
                    self._testSet[idx][i][j] = feature
                    # print(feature)
            idx += 1

    def loadTest(self, X_test, y_test):
        idx = 0
        for label in y_test:
            self._y_test[idx] = int(label.strip())

            for i in range(self._disj_row):
                lines = []
                for each in range(self._n):
                    line = X_test.readline()
                    lines.append(line)

                for j in range(self._disj_col):
                    values = []
                    for p in range(self._n):
                        for q in range(self._m):
                            values.append(lines[p][j*self._m + q])

                    feature = tuple(values)
                    self._testSet[idx][i][j] = feature
                    # print(feature)
            idx += 1

    def evaluation(self):
        correct = 0 
        idx = 0
        t_start = time.time()
        for label in self._y_test:

            predict = None
            max_prob = -math.inf
            # assign the image to digit with highest posterior
            for i in range(10):
                prior = math.log2(self._priors[i])
                # calculate logP(wi | class)
                for j in range(self._disj_row):
                    for k in range(self._disj_col):
                        # print(prior)
                        if self._likelihoods[i][j][k].get(self._testSet[idx][j][k]) == None:
                            prior += math.log2(self._k) - math.log2(self._counts[i] + self._k * self._v)
                        else:
                            prior += self._likelihoods[i][j][k][self._testSet[idx][j][k]]

                if prior > max_prob:
                    max_prob = prior
                    predict = i
            
            if predict == label:
                correct += 1
            # print(correct)
            idx += 1
        t_end = time.time()
        testing_time = t_end - t_start
        print("Testing time is: %fs" % testing_time)

        correct_rate = correct / idx
        print('Disjoint Patches: ', self._n, '*', self._m)
        print('Correct percentage for test images: ', correct_rate)

    def evaluation_overlap(self):
        correct = 0 
        idx = 0
        t_start = time.time()
        for label in self._y_test:

            predict = None
            max_prob = -math.inf
            # assign the image to digit with highest posterior
            for i in range(10):
                prior = math.log2(self._priors[i])
                # calculate logP(wi | class)
                for j in range(self._overlap_row):
                    for k in range(self._overlap_col):
                        # print(prior)
                        if self._likelihoods_overlap[i][j][k].get(self._testSet[idx][j][k]) == None:
                            prior += math.log2(self._k) - math.log2(self._counts[i] + self._k * self._v)
                        else:
                            prior += self._likelihoods_overlap[i][j][k][self._testSet[idx][j][k]]

                if prior > max_prob:
                    max_prob = prior
                    predict = i
            
            if predict == label:
                correct += 1
            # print(correct)
            idx += 1
        t_end = time.time()
        testing_time = t_end - t_start
        print("Testing time is: %fs" % testing_time)
        correct_rate = correct / idx
        print('Overlapping Patches: ', self._n, '*', self._m)
        print('Correct percentage for test images: ', correct_rate)



if __name__ == '__main__':

    # k from 0.1 to 10
    k = 0.1

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(2, 2)
    classifier.train(X_train, y_train, k)
    classifier.loadTest(X_test, y_test)
    classifier.evaluation()
    print()

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(2, 4)
    classifier.train(X_train, y_train, k)
    classifier.loadTest(X_test, y_test)
    classifier.evaluation()
    print()

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(4, 2)
    classifier.train(X_train, y_train, k)
    classifier.loadTest(X_test, y_test)
    classifier.evaluation()
    print()

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(4, 4)
    classifier.train(X_train, y_train, k)
    classifier.loadTest(X_test, y_test)
    classifier.evaluation()
    print()


    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(2, 2)
    classifier.train_overlap(X_train, y_train, k)
    classifier.loadTest_overlap(X_test, y_test)
    classifier.evaluation_overlap()
    print()

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(2, 4)
    classifier.train_overlap(X_train, y_train, k)
    classifier.loadTest_overlap(X_test, y_test)
    classifier.evaluation_overlap()
    print()

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(4, 2)
    classifier.train_overlap(X_train, y_train, k)
    classifier.loadTest_overlap(X_test, y_test)
    classifier.evaluation_overlap()
    print()

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(4, 4)
    classifier.train_overlap(X_train, y_train, k)
    classifier.loadTest_overlap(X_test, y_test)
    classifier.evaluation_overlap()
    print()

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(2, 3)
    classifier.train_overlap(X_train, y_train, k)
    classifier.loadTest_overlap(X_test, y_test)
    classifier.evaluation_overlap()
    print()

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(3, 2)
    classifier.train_overlap(X_train, y_train, k)
    classifier.loadTest_overlap(X_test, y_test)
    classifier.evaluation_overlap()
    print()

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')
    classifier = DigitClassifier(3, 3)
    classifier.train_overlap(X_train, y_train, k)
    classifier.loadTest_overlap(X_test, y_test)
    classifier.evaluation_overlap()

