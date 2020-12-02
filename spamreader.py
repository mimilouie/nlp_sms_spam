##
 # Mimi Louie & Hava Parker
 # CS159 Final Project 
##

from string import punctuation
from scipy import sparse
from collections import Counter
import argparse
import re
import sys
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import cross_val_predict

class BagOfWords():
    def __init__(self, fp, lower=True, bigrams=False, trigrams=False):
        self.labels = []
        self.messages = []
        self.features = []
        self.vocab = [] # add when we filter frequent/infreq words out of features
        self.lower = lower
        self.bigrams = bigrams
        self.trigrams = trigrams

        self.readLabels(fp)
        self.readMessages(fp)

    def readLabels(self, fp):
        lines = fp.readlines()
        labels = []
        for l in lines:
            if l[:3] == "ham":
                labels.append("ham")
            elif l[:4] == "spam":
                labels.append("spam")
        fp.seek(0)
        self.labels = labels

    def readMessages(self, fp):
        lines = fp.readlines()
        messages = []
        for l in lines:
            if l[:3] == "ham":
                messages.append(l[4:-1])
            elif l[:4] == "spam":
                messages.append(l[5:-1])
        fp.seek(0)
        self.messages = messages

    def getBigrams(self, msgs):
        bigrams = []
        for msg in msgs:
            #words = msg.split()
            words = re.findall('[a-zA-Z]+|[0-9]+|[{0}]+'.format(punctuation), msg)
            for i in range(len(words) - 1):
                bigram = words[i] + "-" + words[i+1]
                #bigrams.append(bigram.lower()) if lower else bigrams.append(bigram)
                if self.lower:
                    bigrams.append(bigram.lower())
                else:
                    bigrams.append(bigram)
        return Counter(bigrams)

    def getTrigrams(self, msgs):
        trigrams = []
        for msg in msgs:
            #words = msg.split()
            words = re.findall('[a-zA-Z]+|[0-9]+|[{0}]+'.format(punctuation), msg)
            for i in range(len(words) - 2):
                trigram = words[i] + "-" + words[i+1] + "-" + words[i+2]
                #trigrams.append(trigram.lower()) if lower else trigrams.append(trigram)
                if self.lower:
                    trigrams.append(trigram.lower())
                else:
                    trigrams.append(trigram)
        return Counter(trigrams)

    def makeVocab(self, start=None, end=None):
        words = []
        for msg in self.messages:
            #msgWords = msg.split() #(r'[.!]+|[\w]+')
            # different tokenization approaches l8r
            msgWords = re.findall('[a-zA-Z]+|[0-9]+|[{0}]+'.format(punctuation), msg)
            #print(msgWords)
            for word in msgWords: 
                # <expression1> if <condition> else <expression2>
                #print(word)
                #words.append(word.lower()) if lower else words.append(word)
                if self.lower:
                    words.append(word.lower())
                else:
                    words.append(word)
        #print(words)
        wordCounter = Counter(words)
        sortedWords = wordCounter.most_common()
        wordList = [t[0] for t in sortedWords]
        #self.vocab = wordList[start:end]
        self.vocab = wordList
        #print(wordList[start:end])
    
    # add vocab
    # ngrams, different tokenization, split after punctuation (. but they might be tricky bc of links and stuff idk ! ?)
    # capitalization
    def makeFeatures(self, start=None, end=None):
        self.makeVocab(start, end)
        features = self.vocab
        if self.bigrams:
            features += list(self.getBigrams(self.messages).keys())
        if self.trigrams:
            features += list(self.getTrigrams(self.messages).keys())
        self.features = features

    def extractFeatures(self, msg):
        cnt = Counter(msg.split()) #change to some tokenization strat later idk
        if self.bigrams:
            cnt += self.getBigrams([msg])
        if self.trigrams:
            cnt += self.getTrigrams([msg])
        msgVector = [0]*len(self.features)
        for i in range(len(self.features)):
            if self.features[i] in cnt:
                msgVector[i] = cnt[self.features[i]]
        return msgVector

    #   features -> word1, word2, ...
    #   msg1
    #   msg2
    #   ...
    def process(self):
        N = len(self.messages)
        M = len(self.features)
        X = sparse.lil_matrix((N, M), dtype='uint8')
        for i in range(N):
            if i % 75 == 0:
                print("message #", i)
            msgVector = self.extractFeatures(self.messages[i])
            for j in range(M):
                X[i,j] = msgVector[j]
        return X

def main(args):
    bagOfWords = BagOfWords()
    msgs = readMessages(args.data)
    bagOfWords.readLabels(args.data)
    bagOfWords.makeFeatures(msgs)
    # dont forget parameters
    X = bagOfWords.process(msgs)
    print(X.toarray())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=argparse.FileType('r'), help="Dataset text file")

    args=parser.parse_args()
    main(args)

    args.data.close()