import argparse
import spamreader as sr
import sys
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import cross_val_predict

def main(args):
    # use vocab when we filter the features blah blah blah
    bagOfWords = sr.BagOfWords(args.training, args.lower, args.trigrams, args.bigrams)
    print("msgs [main]: \n", bagOfWords.messages)
    bagOfWords.makeFeatures(args.start, args.end)
    clf = MultinomialNB()
    X = bagOfWords.process()
    
    # crossval for now
    probabilities_validate = cross_val_predict(clf,X,y=bagOfWords.labels,method="predict_proba",cv=args.xvalidate)
    predict_validate = cross_val_predict(clf,X,y=bagOfWords.labels,method="predict",cv=args.xvalidate)
    for i in range(len(bagOfWords.messages)):
        p = predict_validate[i]
        args.output.write(str(i) + " " + p + " " + str(probabilities_validate[i]) + "\n")

    print(predict_validate, "\n")
    print(bagOfWords.features)

# def main(args):
#     bagOfWords = sr.BagOfWords(args.training, args.lower, args.bigrams, args.trigrams)
#     print("MSGS:\n", bagOfWords.messages)
#     print("LABELS:\n", bagOfWords.labels)
#     bagOfWords.makeFeatures(args.start, args.end)
#     # dont forget parameters
#     X = bagOfWords.process()
#     print(X.toarray())

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
    main(args)

    args.training.close()
    args.output.close()