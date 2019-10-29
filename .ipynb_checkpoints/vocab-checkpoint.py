

import torch
import os
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import itertools
import math
import pandas as pd
import unicodedata
import codecs
import itertools



import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np

import  pprint
pp = pprint.PrettyPrinter(indent = 4)



# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

EOC_token = 3 # end of correct answer

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {'eoc': EOC_token}
        self.word2count = {'eoc': 0}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", EOC_token: "eoc"}
        self.num_words = 4  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold # CHANGE probably shouldn't do this
    def trim(self, min_count):
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)        
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", EOC_token: "eoc"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)
            
# make data simple
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def simplify_prep(s):
    rv = s.split()[1]
    if rv == 'di' or rv == 'in':
        return s.split()[2]
    else:
        return rv
    
def simplify_color(s):
    return s

def simplify_obj(s):
    s = normalizeString(s)   
    if len(s.split()) > 1:
        return s.split()[1]
    else:
        return s

#only keep first x exchanges, return pastconvo
def shorten_pastconvo(pastconvo_one, exchanges):
    numexchange = pastconvo_one.count('Tutor:')
    if numexchange > exchanges:
        return "Tutor:".join(pastconvo_one.split("Tutor:", exchanges+1)[:exchanges+1])
    else:
        return pastconvo_one

#added correct and translations
def construct_pastconvo(xy, exchanges, translations):
    pastconvo = []
    for i in range(len(xy)):
        pastconvo_one = shorten_pastconvo(xy['Past Convo'][i], exchanges)
        p = xy['Prep'][i]
        o = xy['Obj'][i]
        c = xy['Color'][i]
        eoc = 'EOC'
        pt = translations[simplify_prep(p)]
        ot = translations[simplify_obj(o)]
        ct = translations[simplify_color(c)]
        
        pastconvo.append(' '.join([p,o,c, pt,ot,ct, eoc, pastconvo_one]))
    return pastconvo
    
