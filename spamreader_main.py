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

# def main(args):
#     # use vocab when we filter the features blah blah blah
#     bagOfWords = sr.BagOfWords()
#     f = open("150.txt",'r')
#     print(f)
#     g = open("output-lowered-bigrams-2.txt",'w')
#     labels = readLabels(f)
#     msgs = sr.readMessages(f)
#     print("msgs [main]: \n", msgs)
#     print("fread\n", f.read())
#     #print("argsread\n", args.training.read())
#     bagOfWords.makeFeatures(msgs, args.lower, args.start, args.end, args.bigrams, args.trigrams)
#     clf = MultinomialNB()
#     X = bagOfWords.process(msgs, args.lower, args.bigrams, args.trigrams)
    
#     # crossval for now
#     probabilities_validate = cross_val_predict(clf,X,y=labels,method="predict_proba",cv=args.xvalidate)
#     predict_validate = cross_val_predict(clf,X,y=labels,method="predict",cv=args.xvalidate)
#     for i in range(len(msgs)):
#         p = predict_validate[i]
#         g.write(str(i) + " " + p + " " + str(probabilities_validate[i]) + "\n")

def main(args):
    bagOfWords = sr.BagOfWords()
    msgs = sr.readMessages(args.training)
    readLabels(args.training)
    bagOfWords.makeFeatures(msgs, args.lower, args.end, args.bigrams, args.trigrams)
    # dont forget parameters
    X = bagOfWords.process(msgs, args.lower, args.bigrams, args.trigrams)
    print(X.toarray())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=argparse.FileType('r'), help="training dataset filepath")
    parser.add_argument("-o", "--output", type=argparse.FileType('w'), default=sys.stdout, help="prediction writing location")
    parser.add_argument("-x", "--xvalidate", type=int, default=10, help="crossvalidation folds")
    parser.add_argument("-l", "--lower", action="store_true", default=False, help="use lowercase")
    parser.add_argument("-b", "--bigrams", action="store_true", default=False, help="use bigrams")
    parser.add_argument("-t", "--trigrams", action="store_true", default=False, help="use trigrams")
    parser.add_argument("-s", "--start", type=int, default=None, help="begin of vocab cutoff")
    parser.add_argument("-e", "--end", type=int, default=None, help="end of vocab cutoff")

    args = parser.parse_args()
    print(args)
    main(args)

    # args.training.close()
    # args.output.close()