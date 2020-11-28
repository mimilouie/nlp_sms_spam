"""Calculates the measures for the PAN19 hyperpartisan news detection task"""
# Modifications include: argparse, warning, division by zero handling

import argparse
import json
import os
import sys
import warnings
from collections import Counter

def main(args):
    groundTruth = {}

    lines = args.inputDataset.readlines()
    labels = []
    for i in range(len(lines)):
        if lines[i][:3] == "ham":
            groundTruth[str(i)] = "ham"
        elif lines[i][:4] == "spam":
            groundTruth[str(i)] = "spam"
            
    c = Counter()

    for line in args.inputRun: 
        values = line.rstrip('\n').split()
        messageId, prediction = values[:2]
        c[(prediction, groundTruth[messageId])] += 1
  
    if sum(c.values()) < len(groundTruth):
        warnings.warn("Missing {} predictions".format(len(groundTruth) - sum(c.values())), UserWarning)
        
    tp = c[('spam', 'spam')]
    tn = c[('ham', 'ham')]
    fp = c[('spam', 'ham')]
    fn = c[('ham', 'spam')]
        
    accuracy  = (tp + tn) / sum(c.values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {"truePositives": tp, "trueNegatives": tn, "falsePositives": fp, "falseNegatives": fn,
               "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    json.dump(results, args.outputFile, indent=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--inputDataset", type=argparse.FileType('r'), required=True)
    parser.add_argument("-r", "--inputRun", type=argparse.FileType('r'), required=True)
    parser.add_argument("-o", "--outputFile", type=argparse.FileType('w'), default=sys.stdout)

    args=parser.parse_args()

    main(args)

    args.outputFile.close()
