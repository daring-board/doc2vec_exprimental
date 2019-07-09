import pandas as pd
from gensim import models
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
        # print(token)
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

if __name__ == '__main__':
    tokenizer = Tokenizer(mmap=True)
    char_filters = [UnicodeNormalizeCharFilter()]
    token_filters = [POSStopFilter(['記号','助詞']), LowerCaseFilter()]
    analyzer = Analyzer(char_filters, tokenizer, token_filters)

    model_path = './models/doc2vec.model'
    delim = '_'

    programs = pd.read_pickle('data/example/programs.pkl')
    p_text = {'prog_%d'%key: programs[key]['text'] for key in programs.keys()}

    creatives = pd.read_pickle('data/example/creatives.pkl')
    c_text = {'crea_%d'%key: (creatives[key]['text'], creatives[key]['creative_category']) for key in creatives.keys()}

    p_id = list(p_text.keys())[100]
    # print(split_into_words(p_text[p_id], analyzer))
    print(p_text[p_id])
    # print(p_text[p_id][1])
    print()
    print()
    print()

    model = models.Doc2Vec.load(model_path)
    syms = model.docvecs.most_similar(p_id, topn=15)
    # print(model.docvecs[0])

    for s_id in syms:
        prefix = s_id[0].split(delim)[0]
        if prefix == 'prog':
            continue
            # print(s_id[1])
            # print(p_text[s_id[0]])
            # print(p_text[s_id[0]][1])
        else:
            print(s_id[1])
            print(c_text[s_id[0]][0])
            print(c_text[s_id[0]][1])
            # print(split_into_words(c_text[s_id[0]], analyzer))
            print()

