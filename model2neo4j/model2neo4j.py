import logging
import os
import sys
from collections import namedtuple

import boto3
import pandas as pd
from gensim import matutils
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from py2neo import Graph, Node, NodeMatcher, Relationship

logging.basicConfig(level=logging.INFO)


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
                logging.info("Created similarity for paper id: " + arxiv_id)


if __name__ == "__main__":
    '''
    S3 Configuration, all the models and data will be saved to S3. Roles and policy need to configured in AWS for S3.
    '''
    s3 = boto3.resource('s3')
    S3_BUCKET = 'researchkernel-machinelearning'
    S3_BUCKET = 'researchkernel-machinelearning'
    MODEL_KEY = 'models/arxivAbstractModel'
    VECTORS_KEY = 'models/arxivAbstractModel.docvecs.vectors_docs.npy'
    SYN1NEG_KEY = 'models/arxivAbstractModel.trainables.syn1neg.npy'
    WV_VECTORS_KEY = 'models/arxivAbstractModel.wv.vectors.npy'
    try:
        #  For testing purpose only
        # filepath = sys.argv[1]
        # data = pd.read_csv(filepath)
        # print(data.head(5))
        # url = sys.argv[2]
        # port = sys.argv[3]
        '''
        Taking input from environment variables, easier for running on docker based jobs.
        '''
        filepath = os.environ['filepath']
        data = pd.read_csv(filepath)
        url = os.environ['neo4j_endpoint']
        port = os.environ['neo4j_port']

        graph = Graph()
        s3.Bucket(S3_BUCKET).download_file(
            MODEL_KEY, './models/arxivAbstractModel')
        s3.Bucket(S3_BUCKET).download_file(
            VECTORS_KEY, './models/arxivAbstractModel.docvecs.vectors_docs.npy')
        s3.Bucket(S3_BUCKET).download_file(
            SYN1NEG_KEY, './models/arxivAbstractModel.trainables.syn1neg.npy')
        s3.Bucket(S3_BUCKET).download_file(
            WV_VECTORS_KEY, './models/arxivAbstractModel.wv.vectors.npy')
    except Exception as e:
        print(e)
        logging.info(e)
    try:
        model = Doc2Vec.load('./models/arxivAbstractModel')
        neo_relationship_creator([data], [model], [graph])
    except Exception as e:
        print(e)
        logging.info(e)
