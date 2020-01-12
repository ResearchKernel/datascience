import json
import logging
import os
import sys
import boto3
import random
import time
from collections import namedtuple
from multiprocessing import Pool

import pandas as pd
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from clean_metadata import (clean_df, parallelize_dataframe)
logging.basicConfig(level=logging.INFO)
'''
Reading textfiles from folder and making dataframe from it
'''


def Doc2vec_traning(dataframe):
    tagged_docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for text, tags in zip(dataframe['content'].tolist(), dataframe['arxiv_id'].tolist()):
        tags = [tags]
        tagged_docs.append(analyzedDocument(text, tags))
    model = Doc2Vec(size=400  # Model initialization
                    , window=5, min_count=5, workers=4)
    model.build_vocab(tagged_docs)  # Building vocabulary
    alpha_val = 0.025        # Initial learning rate
    min_alpha_val = 1e-4     # Minimum for linear learning rate decay
    passes = 2              # Number of passes of one document during training
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

    print("Model Saved to folder")
    return model


if __name__ == "__main__":
    '''
    S3 Configuration, all the models and data will be saved to S3. Roles and policy need to configured in AWS for S3.
    '''
    s3 = boto3.resource('s3')
    S3_BUCKET = 'researchkernel-machinelearning'
    MODEL_KEY = 'models/arxivAbstractModel'
    VECTORS_KEY = 'models/arxivAbstractModel.docvecs.vectors_docs.npy'
    SYN1NEG_KEY = 'models/arxivAbstractModel.trainables.syn1neg.npy'
    WV_VECTORS_KEY = 'models/arxivAbstractModel.wv.vectors.npy'
    try:
        # Read new data
        logging.info("Reading file...")
        data = os.environ['filepath']
        data['content'] = data['title'] + data['abstract']
        logging.info("complete reading file... :)")
        #  data cleaning
        data = parallelize_dataframe(data, clean_df)
        logging.info("Cleaning complete")
        model = Doc2vec_traning(data)
        logging.info("Staring Traning ...")
        # upload new trained model back to S3
        logging.info("Saving model to s3")
        s3.Bucket(S3_BUCKET).upload_file(
            "./models/arxivAbstractModel", MODEL_KEY)
        s3.Bucket(S3_BUCKET).upload_file(
            "./models/arxivAbstractModel.docvecs.vectors_docs.npy", VECTORS_KEY)
        s3.Bucket(S3_BUCKET).upload_file(
            "./models/arxivAbstractModel.trainables.syn1neg.npy", SYN1NEG_KEY)
        s3.Bucket(S3_BUCKET).upload_file(
            "./models/arxivAbstractModel.wv.vectors.npy", WV_VECTORS_KEY)
        logging.info("Model saved to S3.")
    except Exception as e:
        print(e)
        logging.info(e)
