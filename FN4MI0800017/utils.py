#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
##########################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import random
import nltk
import json
import time
from nltk.translate.bleu_score import corpus_bleu
from torch.nn import parameter
nltk.download('punkt')

from tokenizers import Tokenizer

from parameters import *

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")

def encode(text, tokens, token_list, unkToken):
    encoded_text = []
    i = 0
    while i < len(text):
        match = None
        for token in token_list:
            if text[i : i + len(token)] == token:
                match = token
                break
        if match:
            # encoded_text.append(tokens[match])
            encoded_text.append(match)
            i += len(match)
        elif text[i] == "\n":
            i += 1
        else:
            # encoded_text.append(tokens[unkToken])
            encoded_text.append(unkToken)
            i += 1
    return encoded_text

def readCorpus(fileName):
    print('Loading file:',fileName)

    tokenizer = Tokenizer.from_file(tokensFileName)
    return [ tokenizer.encode(line).tokens for line in open(fileName, encoding="utf-8") ]
    
    # encoded_sentences = []

    # with open(fileName, encoding="utf-8") as file:
    #     lines = file.readlines()
    #     total_lines = len(lines)

    #     start_time = time.time()
        
    #     for i, line in enumerate(lines):
    #         encoded_sentences.append(encode(line, tokens, token_list, unkToken))

    #         if (i + 1) % 50 == 0:
    #             elapsed_time = time.time() - start_time
    #             avg_time_per_line = elapsed_time / (i + 1)
    #             remaining_time = avg_time_per_line * (total_lines - (i + 1))
    #             hours, rem = divmod(remaining_time, 3600)
    #             minutes, seconds = divmod(rem, 60)
    #             print(f'Encoded: {i + 1}/{total_lines} - Estimated finish in {int(hours):02}:{int(minutes):02}:{int(seconds):02}')

    # return encoded_sentences

def getDictionary(corpus, startToken, endToken, unkToken, padToken, transToken, wordCountThreshold = 2):
    dictionary={}
    for s in corpus:
        for w in s:
            if w in dictionary: dictionary[w] += 1
            else: dictionary[w]=1

    words = [startToken, endToken, unkToken, padToken, transToken] + [w for w in sorted(dictionary) if dictionary[w] > wordCountThreshold]
    print(len(words))
    print(list(words)[:5])
    return { w:i for i,w in enumerate(words)}


def prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken):

    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)
    word2ind = getDictionary(sourceCorpus+targetCorpus, startToken, endToken, unkToken, padToken, transToken)

    print(f"Dictionary length: {len(list(word2ind))}")
    print(list(word2ind)[:10])

    trainCorpus = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceCorpus,targetCorpus)]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    devCorpus = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceDev,targetDev)]

    print('Corpus loading completed.')
    return trainCorpus, devCorpus, word2ind
