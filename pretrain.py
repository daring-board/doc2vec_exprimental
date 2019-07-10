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

def corpus_files():
    base_path = './data/text'
    dirs = [path.join(base_path, x) for x in listdir(base_path) if not x.endswith('.txt')]
    docs = [path.join(x, y) for x in dirs for y in listdir(x) if not x.startswith('LICENSE')]
    return docs

def read_document(path):
    with open(path, 'r', encoding='utf8') as f: return f.read()

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
    def __init__(self, corpus):
        docs = [read_document(x) for x in corpus]
        self.obj = list(zip(docs, corpus))
        self.current = 0
        tokenizer = Tokenizer(mmap=True)
        char_filters = [UnicodeNormalizeCharFilter()]
        token_filters = [POSStopFilter(['記号','助詞']), LowerCaseFilter()]
        self.tokenizer = Analyzer(char_filters, tokenizer, token_filters)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current == len(self.obj):
            raise StopIteration()
        (doc, name) = self.obj[self.current]
        # print(name)
        sys.stdout.write('\r前処理中 {}/{}'.format(self.current, len(self.obj)))
        self.current += 1
        return doc_to_sentence(doc, name.split('\\')[-1], self.tokenizer)

if __name__ == "__main__":
    corpus = corpus_files()
    sentences = corpus_to_sentences(corpus)

    model = models.Doc2Vec(sentences, dm=0, size=300, window=10, workers=6)
    model.save('./models/pretrain/doc2vec.model')
