import numpy as np
import pandas as pd
from gensim import models
import pprint
import umap
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.externals import joblib

if __name__ == '__main__':

    data = [row.strip().split(',') for row in open('data/example/program_group_labels.csv', 'r')][1:]
    label = {int(row[0]): int(row[2]) for row in data}

    programs = pd.read_pickle('data/example/programs.pkl')
    p_text = {'prog_%d'%key: programs[key]['text'] for key in programs.keys()}
    creatives = pd.read_pickle('data/example/creatives.pkl')
    c_text = {'crea_%d'%key: (creatives[key]['text'], creatives[key]['creative_category']) for key in creatives.keys()}

    model_path = './models/doc2vec.model'
    # model_path = './models/pretrain/retrained_doc2vec.model'
    model = models.Doc2Vec.load(model_path)

    vector_list = [model.docvecs[key] for key in p_text.keys()]
    label_list = [label[int(key.split('_')[-1])] for key in p_text.keys()]

    vector_list = np.array(vector_list)
    label_list = np.array(label_list)

    reducer = umap.UMAP(n_neighbors=10, n_components=25)
    embedding = reducer.fit_transform(vector_list, y=label_list)
    joblib.dump(reducer, './models/reducer.sav')

    classifier = GBC()
    classifier.fit(embedding, label_list)
    joblib.dump(classifier, './models/classifier.sav')
    
    test_vectors = [model.docvecs[key] for key in p_text.keys()]
    test_vectors = reducer.transform(test_vectors)
    predict = classifier.predict(test_vectors)

    count = 0
    with open('./data/example/result.csv', 'w', encoding='sjis') as f:
        f.write('id,estimate_program_group\n')
        for key in p_text.keys():
            f.write('%s,%d\n'%(key.split('_')[-1], predict[count]))
            count += 1
    
