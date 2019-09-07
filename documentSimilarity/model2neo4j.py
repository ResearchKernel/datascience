from py2neo import Graph, Node, Relationship, NodeMatcher
from collections import namedtuple
from gensim import matutils
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def neo_relationship_creator(dataframe, models, graph):
    '''
    Fucntion takes dataframe, models and graph.
    Create relatioships with existing node
    '''
    for dataframe, model, graph in zip(dataframe, models, graph):
        matcher = NodeMatcher(graph)

        for arxiv_id in dataframe['arxiv_id']:
            similarity = model.docvecs.most_similar(arxiv_id)
            for j, k in similarity:
                paper = matcher.match(
                    'paper', arxiv_id=arxiv_id).first()
                paper_similar = matcher.match('paper', arxiv_id=j).first()
                graph.create(Relationship(
                    paper, "SIMILAR_TO", paper_similar, score=k))


if __name__ == "__main__":
    model = Doc2Vec.load('../models/arxiv_full_text')
    graph = Graph()
    data = pd.read_csv("../data/2017-01-01full.csv")
    neo_relationship_creator([data.head(1000)], [model], [graph])
