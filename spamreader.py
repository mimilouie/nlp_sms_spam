##
 # Mimi Louie & Hava Parker
 # CS159 Final Project 
##

from scipy import sparse
from collections import Counter
import argparse

class BagOfWords():
    def __init__(self):
        self.features = []
        self.vocab = [] # add when we filter frequent/infreq words out of features
        self.labels = []

    def readMessages(self, fp):
        lines = fp.readlines()
        messages = []
        for l in lines:
            if l[:4] == "ham\t":
                messages.append(l[4:-1])
            elif l[:5] == "spam\t":
                messages.append(l[5:-1])
        return messages

    def makeDict(self, msgs):
        words = []
        for msg in msgs:
            msgWords = msg.split() #(r'[.!]+|[\w]+')
            # different tokenization approaches l8r
            for word in msgWords: 
                words.append(word.lower())
                # lowercased for now
        cnt = Counter(words)
        # keeping counter for now; we can filter by
        # feature frequency/infrequency later
        self.features = list(cnt.keys())

    def extractFeatures(self, features, msg):
        counts = Counter(msg.split())
        msgVector = [0]*len(features)
        for i in range(len(features)):
            if features[i] in counts:
                msgVector[i] = counts[features[i]]
        return msgVector

    #   features -> word1, word2, ...
    #   msg1
    #   msg2
    #   ...
    def process(self, features, msgs):
        N = len(msgs)
        M = len(features)
        X = sparse.lil_matrix((N, M), dtype='uint8')
        for i in range(100): #range(N):
            print("message #", i)
            msgVector = self.extractFeatures(features, msgs[i])
            for j in range(M):
                X[i,j] = msgVector[j]
        return X

def main(args):
    bagOfWords = BagOfWords()
    msgs = bagOfWords.readMessages(args.data)
    bagOfWords.readLabels(args.data)
    bagOfWords.makeDict(msgs)
    X = bagOfWords.process(bagOfWords.features, msgs)
    print(X.toarray())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=argparse.FileType('r'), help="Dataset text file")

    args=parser.parse_args()
    main(args)

    args.data.close()