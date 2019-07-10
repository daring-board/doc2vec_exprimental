import numpy as np
import pandas as pd
from gensim import models
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    data = [row.strip().split(',') for row in open('data/example/program_group_labels.csv', 'r')][1:]
    label = {int(row[0]): int(row[2]) for row in data}

    programs = pd.read_pickle('data/example/programs.pkl')
    p_text = {'prog_%d'%key: programs[key]['text'] for key in programs.keys()}

    model_path = './models/doc2vec.model'
    model = models.Doc2Vec.load(model_path)

    vector_list = [model.docvecs[key] for key in p_text.keys()]
    label_list = [label[int(key.split('_')[-1])] for key in p_text.keys()]

    vector_list = np.array(vector_list)
    label_list = np.array(label_list)

    reducer_path = './models/reducer.sav'
    reducer = joblib.load(reducer_path)
    embedding = reducer.transform(vector_list)

    classifier_path = './models/classifier.sav'
    classifier = joblib.load(classifier_path)
    predicts = classifier.predict(embedding)

    pca = PCA(n_components=3)
    pca.fit(embedding)
    features = pca.transform(embedding)

    count = 0
    label_num = {v: 0 for k, v in label.items()}
    unit =  int(256 / len(label_num))
    for key in label_num.keys():
        label_num[key] = count
        count += 1

    fig = plt.figure()
    ax = Axes3D(fig)

    for idx in range(len(predicts)):
        r = label_num[predicts[idx]] * unit
        h_r = ('%0s'%hex(r)[2:]).zfill(2)
        b = 255 - label_num[predicts[idx]] * unit
        h_b = ('%0s'%hex(b)[2:]).zfill(2)
        color = '#%sAA%s'%(h_r, h_b)
        ax.scatter(features[idx, 0], features[idx, 1], features[idx, 2], "o", color=color)
        # plt.scatter(features[idx, 0], features[idx, 1], c=color)
    plt.show()

    
