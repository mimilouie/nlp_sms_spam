import argparse
import spamreader as sr
import sys
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import cross_val_predict

def readLabels(fp):
    lines = fp.readlines()
    labels = []
    for l in lines:
        if l[:3] == "ham":
            labels.append("ham")
        elif l[:4] == "spam":
            labels.append("spam")
    return labels

def doExperiment(args):
    # use vocab when we filter the features blah blah blah
    labels = readLabels(args.training)
    bagOfWords = sr.BagOfWords()
    f = open("150.txt",'r')
    g = open("output-notlowered-bigrams.txt",'w')
    msgs = bagOfWords.readMessages(f)
    bagOfWords.makeFeatures(msgs)
    clf = MultinomialNB()
    X = bagOfWords.process(bagOfWords.features, msgs)
    # crossval for now
    probabilities_validate = cross_val_predict(clf,X,y=labels,method="predict_proba",cv=args.xvalidate)
    predict_validate = cross_val_predict(clf,X,y=labels,method="predict",cv=args.xvalidate)
    for i in range(len(msgs)):
        p = predict_validate[i]
        g.write(str(i) + " " + p + " " + str(probabilities_validate[i]) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type=argparse.FileType('r'), help="training dataset filepath")
    parser.add_argument("-o", "--output", type=argparse.FileType('w'), default=sys.stdout, help="prediction writing location")
    parser.add_argument("-x", "--xvalidate", type=int, default=10, help="crossvalidation folds")

    args = parser.parse_args()
    doExperiment(args)

    args.training.close()