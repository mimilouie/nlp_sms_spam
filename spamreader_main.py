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
        if l[:4] == "ham\t":
            labels.append("ham")
        elif l[:5] == "spam\t":
            labels.append("spam")
    return labels

def doExperiment(args):
    # use vocab when we filter the features blah blah blah
    labels = readLabels(args.training)
    return labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=argparse.FileType('r'), help="training dataset filepath")
    parser.add_argument("-o", "--output", type=argparse.FileType('w'), default=sys.stdout, help="prediction writing location")

    args = parser.parse_args()
    doExperiment(args)

    args.training.close()