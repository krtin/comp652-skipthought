# Evaluation for MSRP

import numpy as np

from collections import defaultdict
from nltk.tokenize import word_tokenize
from numpy.random import RandomState
import os.path
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as f1


def evaluate(encoder, k=10, seed=3456, evalcv=True, evaltest=False, loc='./data/'):
    print 'Load Data...'
    traintext, testtext, labels = load_data(loc)

    print 'Convert to sentence embeddings...'

    trainA = encoder.encode(traintext[0], verbose=False)
    trainB = encoder.encode(traintext[1], verbose=False)

    if evalcv:
        print 'Perform cross-validation...'
        C = eval_kfold(trainA, trainB, traintext, labels[0], shuffle=True, k=10, seed=3456)
    #print("Size of sentences: ",trainA.shape)
    if evaltest:
        if not evalcv:
            C = 4    

        print 'Convert test data to skipthought vectors...'
        testA = encoder.encode(testtext[0], verbose=False)
        testB = encoder.encode(testtext[1], verbose=False)

	#u.v and u-v features concatenation 
        train_features = np.c_[np.abs(trainA - trainB), trainA * trainB]
        test_features = np.c_[np.abs(testA - testB), testA * testB]

        print 'Evaluate logistic regression...'
        clf = LogisticRegression(C=C)
	#fit model
        clf.fit(train_features, labels[0])
        #get prediction
	ypred = clf.predict(test_features)
        print 'Test accuracy: ' + str(clf.score(test_features, labels[1]))
        #get f1 score, label 1 is true value
	print 'Test F1: ' + str(f1(labels[1], ypred))


def load_data(loc='./data/'):
    trainloc = os.path.join(loc, 'msr_paraphrase_train.txt')
    testloc = os.path.join(loc, 'msr_paraphrase_test.txt')

    trainA, trainB, testA, testB = [],[],[],[]
    trainS, devS, testS = [],[],[]

    f = open(trainloc, 'rb')
    for line in f:
        text = line.strip().split('\t')
        trainA.append(' '.join(word_tokenize(text[3])))
        trainB.append(' '.join(word_tokenize(text[4])))
        trainS.append(text[0])
    f.close()
    f = open(testloc, 'rb')
    for line in f:
        text = line.strip().split('\t')
        testA.append(' '.join(word_tokenize(text[3])))
        testB.append(' '.join(word_tokenize(text[4])))
        testS.append(text[0])
    f.close()

    trainS = [int(s) for s in trainS[1:]]
    testS = [int(s) for s in testS[1:]]

    return [trainA[1:], trainB[1:]], [testA[1:], testB[1:]], [trainS, testS]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def eval_kfold(A, B, train, labels, shuffle=True, k=10, seed=3456):
    # features
    labels = np.array(labels)
    features = np.c_[np.abs(A - B), A * B]

    scan = [2**t for t in range(0,9,1)]
    npts = len(features)
    kf = KFold(npts, n_folds=k, shuffle=shuffle, random_state=seed)
    scores = []

    for s in scan:

        scanscores = []

        for train, test in kf:

            # Split data
            X_train = features[train]
            y_train = labels[train]
            X_test = features[test]
            y_test = labels[test]

            # Train classifier
            clf = LogisticRegression(C=s)
            clf.fit(X_train, y_train)
            yhat = clf.predict(X_test)
            fscore = f1(y_test, yhat)
            scanscores.append(fscore)
            print (s, fscore)

        # Append mean score
        scores.append(np.mean(scanscores))
        print scores

    # Get the index of the best score
    s_ind = np.argmax(scores)
    s = scan[s_ind]
    print scores
    print s
    return s


