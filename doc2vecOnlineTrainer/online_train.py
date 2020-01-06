import json
import logging
import sys
import os
import random
import time
from collections import namedtuple
from multiprocessing import Pool

import boto3
import pandas as pd
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from clean_metadata import clean_df, parallelize_dataframe


def online_Doc2vec_traning(dataframe, model):
    tagged_docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for text, tags in zip(dataframe['content'].tolist(), dataframe['filename'].tolist()):
        tags = [tags]
        tagged_docs.append(analyzedDocument(text, tags))
    model.build_vocab(tagged_docs, update=True)  # Building vocabulary
    alpha_val = 0.025        # Initial learning rate
    min_alpha_val = 1e-4     # Minimum for linear learning rate decay
    passes = 15              # Number of passes of one document during training
    alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)

    for epoch in range(passes):

        # Shuffling gets better results

        random.shuffle(tagged_docs)

        # Train

        model.alpha, model.min_alpha = alpha_val, alpha_val

        model.train(tagged_docs, total_examples=model.corpus_count,
                    epochs=model.epochs)

        # Logs

        print('Completed pass %i at alpha %f' % (epoch + 1, alpha_val))

        # Next run alpha

        alpha_val -= alpha_delta

    model.save("./models/arxivAbstractModel")
    return model


def main_online_Doc2vec_traning(model, data):
    model = online_Doc2vec_traning(data, model)


if __name__ == "__main__":
    '''
    S3 Configuration, all the models and data will be saved to S3. Roles and policy need to configured in AWS for S3.
    '''
    s3 = boto3.resource('s3')
    S3_BUCKET = 'researchkernel-machinelearning'
    KEY = 'models/arxivAbstractModel'
    try:
        # Read new data
        data = pd.read_csv(sys.argv[1])
        #  data cleaning
        data = parallelize_dataframe(data, clean_df)
        # Download the model in models folder
        s3.Bucket(S3_BUCKET).download_file(KEY, './models/arxivAbstractModel')
        # Loading model in memory.
        model = Doc2Vec.load('./models/arxivAbstractModel')
        main_online_Doc2vec_traning(model, data)
        # upload new trained model back to S3
        s3.Bucket(S3_BUCKET).upload_file("./models/arxivAbstractModel", KEY)
    except Exception as e:
        print(e)
