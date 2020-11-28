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
            if l[:3] == "ham":
                messages.append(l[4:-1])
            elif l[:4] == "spam":
                messages.append(l[5:-1])
        return messages

    def getBigrams(self, msgs):
        bigrams = []
        for msg in msgs:
            words = msg.split()
            for i in range(len(words) - 1):
                bigram = words[i] + "-" + words[i+1]
                bigrams.append(bigram)
        return bigrams

    def getTrigrams(self, msgs):
        trigrams = []
        for msg in msgs:
            words = msg.split()
            for i in range(len(words) - 2):
                trigram = words[i] + "-" + words[i+1] + "-" + words[i+2]
                trigrams.append(trigram)
        return trigrams

    # add vocab
    # ngrams, different tokenization, split after punctuation (. but they might be tricky bc of links and stuff idk ! ?)
    # capitalization
    def makeFeatures(self, msgs):
        words = []
        for msg in msgs:
            msgWords = msg.split() #(r'[.!]+|[\w]+')
            # different tokenization approaches l8r
            for word in msgWords: 
                words.append(word)
                # lowercased for now
        cnt = Counter(words) + Counter(self.getBigrams(msgs))
        print(cnt)
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
        for i in range(N): #range(N):
            if i % 20 == 0:
                print("message #", i)
            msgVector = self.extractFeatures(features, msgs[i])
            for j in range(M):
                X[i,j] = msgVector[j]
        return X

def main(args):
    bagOfWords = BagOfWords()
    msgs = bagOfWords.readMessages(args.data)
    bagOfWords.readLabels(args.data)
    bagOfWords.makeFeatures(msgs)
    X = bagOfWords.process(bagOfWords.features, msgs)
    print(X.toarray())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=argparse.FileType('r'), help="Dataset text file")

    args=parser.parse_args()
    main(args)

    args.data.close()