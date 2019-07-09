import sys
from os import listdir, path
from janome.tokenizer import Tokenizer
from gensim import models
from gensim.models.doc2vec import LabeledSentence

def corpus_files():
    base_path = './data/text'
    dirs = [path.join(base_path, x) for x in listdir(base_path) if not x.endswith('.txt')]
    docs = [path.join(x, y) for x in dirs for y in listdir(x) if not x.startswith('LICENSE')]
    return docs

def read_document(path):
    with open(path, 'r', encoding='utf8') as f: return f.read()

def split_into_words(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    ret = [token.base_form for token in tokens if token.part_of_speech.split(',')[0] in [u'名詞', u'形容詞']]
    return ret

def doc_to_sentence(doc, name, tokenizer):
    words = split_into_words(doc, tokenizer)
    return LabeledSentence(words=words, tags=[name])

class corpus_to_sentences():
    def __init__(self, corpus):
        docs = [read_document(x) for x in corpus]
        self.obj = list(zip(docs, corpus))
        self.current = 0
        self.tokenizer = Tokenizer(mmap=True)

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

    model = models.Doc2Vec(sentences, dm=0, size=150, window=10, workers=6)
    model.save('doc2vec.model')
