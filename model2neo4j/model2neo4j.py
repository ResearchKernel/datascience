from py2neo import Graph, Node, Relationship, NodeMatcher
from collections import namedtuple
from gensim import matutils
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import logging
import sys

import boto3
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
    '''
    S3 Configuration, all the models and data will be saved to S3. Roles and policy need to configured in AWS for S3.
    '''
    s3 = boto3.resource('s3')
    S3_BUCKET = 'researchkernel-machinelearning'
    KEY = 'models/arxivAbstractModel'
    try:
        data = pd.read_csv(sys.argv[1])
        url = sys.argv[2]
        port = sys.argv[3]
        graph = Graph("http://" + url + ":" + port + "/db/data/")
        s3.Bucket(S3_BUCKET).download_file(KEY, './models/arxivAbstractModel')
    except Exception as e:
        print(e)
        logging.info(e)
    try:
        model = Doc2Vec.load('./models/arxivAbstractModel')
        neo_relationship_creator([data], [model], [graph])
    except Exception as e:
        print(e)
        logging.info(e)
