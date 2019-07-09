import pandas as pd
import numpy as np
import sys
from os import listdir, path
from gensim import models
from gensim.models.doc2vec import LabeledSentence
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
import neologdn
import re

def split_into_words(text, tokenizer):
    # tokens = tokenizer.tokenize(text)
    normalized_text = neologdn.normalize(text)
    normalized_text = re.sub(r'[!-/:-@[-`{-~]', r' ', normalized_text)
    tokens = [token for token in tokenizer.analyze(normalized_text)]

    ret = []
    for idx in range(len(tokens)):
        token = tokens[idx]
        if idx+1 == len(tokens):
            if parts[0] == '名詞' and parts[1] != '接尾' and parts[1] != '副詞可能':
                ret.append(token.base_form)
            elif parts[0] == '名詞': continue
            else:
                ret.append(token.base_form)
            break
        post_token = tokens[idx+1]
        parts = token.part_of_speech.split(',')
        post_parts = post_token.part_of_speech.split(',')
        if parts[0] == '名詞':
            if parts[1] == '一般' and post_parts[0] == '名詞' and post_parts[1] == '接尾':
                ret.append(token.base_form + post_token.base_form)
            elif parts[1] == '一般':
                ret.append(token.base_form)
            elif parts[1] == '接尾': continue
            elif parts[1] == '副詞可能': continue
            else: 
                ret.append(token.base_form)
        else:
            ret.append(token.base_form)
    return ret

def doc_to_sentence(doc, name, tokenizer):
    words = split_into_words(doc, tokenizer)
    return LabeledSentence(words=words, tags=[name])

class corpus_to_sentences():
    def __init__(self, corpus, tokenizer):
        self.current = 0
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.keys = list(corpus.keys())

    def __iter__(self):
        return self

    def __next__(self):
        if self.current == len(self.keys):
            raise StopIteration()
        key = self.keys[self.current]
        sys.stdout.write('\r前処理中 {}/{}'.format(self.current, len(self.keys)))
        item = doc_to_sentence(self.corpus[key], key, self.tokenizer)
        self.current += 1
        return item

if __name__ == '__main__':
    tokenizer = Tokenizer(mmap=True)
    char_filters = [UnicodeNormalizeCharFilter()]
    token_filters = [POSStopFilter(['記号','助詞']), LowerCaseFilter()]
    analyzer = Analyzer(char_filters, tokenizer, token_filters)

    programs = pd.read_pickle('data/example/programs.pkl')
    p_text = {'prog_%d'%key: programs[key]['text'] for key in programs.keys()}

    creatives = pd.read_pickle('data/example/creatives.pkl')
    c_text = {'crea_%d'%key: creatives[key]['text'] for key in creatives.keys()}

    corpus = {}
    corpus.update(p_text)
    corpus.update(c_text)
    sentences = corpus_to_sentences(corpus, analyzer)
    model = models.Doc2Vec(sentences, dm=1, size=150, window=15, workers=6)
    model.save('./models/doc2vec.model')
