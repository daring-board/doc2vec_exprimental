from gensim import models

model = models.Doc2Vec.load('doc2vec.model')

print(model.docvecs.most_similar('dokujo-tsushin-4799933.txt', topn=3))