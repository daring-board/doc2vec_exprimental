import pandas as pd
from gensim import models
import umap
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #主成分分析器
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    model_path = './models/doc2vec.model'
    delim = '_'

    programs = pd.read_pickle('data/example/programs.pkl')
    p_text = {'prog_%d'%key: programs[key]['text'] for key in programs.keys()}

    creatives = pd.read_pickle('data/example/creatives.pkl')
    c_text = {'crea_%d'%key: (creatives[key]['text'], creatives[key]['creative_category']) for key in creatives.keys()}

    model = models.Doc2Vec.load(model_path)
    vectors_list = [model.docvecs[n] for n in range(len(model.docvecs))]

    start_time = time.time()
    embedding = umap.UMAP(n_neighbors=15, n_components=3).fit_transform(vectors_list)
    # embedding = PCA().fit_transform(vectors_list)
    interval = time.time() - start_time

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(embedding[:,0], embedding[:,1], embedding[:, 2])
    plt.show()

    # plt.scatter(embedding[:,0], embedding[:,1])
    # plt.colorbar()
    # plt.savefig('umap.png')


