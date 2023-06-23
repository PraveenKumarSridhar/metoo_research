"""
This is an old component not currently used in the pipeline, but kept around for 
possible future use
"""

from argparse import ArgumentParser

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline 

import pickle
import logging

import numpy as np
import pandas as pd
from pathlib import Path
logger = logging.getLogger()


def news_iden_imp_heu(author,news_id_list):
    auth_heuristics = 'news' in str(author).lower() or author in news_id_list
    return 1 if auth_heuristics else 0

def go(input):
    artifact_path = Path('components/artifacts/')

    data = pd.read_csv(input['input_path'], sep = "\t")
    data["Full Text"] = data["Full Text"].fillna("")

    news_ids = pd.read_csv(r'../data/news_outlets-accounts.csv')
    news_id_list = news_ids['Token'].tolist()

    author_corpus = data.groupby("Author")["Full Text"].apply(" ".join).reset_index()
    author_corpus ['news_label'] = author_corpus['Author'].apply(lambda x: news_iden_imp_heu(x,news_id_list))

    author_corpus = data.groupby("Author")["Full Text"].apply(" ".join)
    
    news_inf = pd.Series(author_corpus.index.str.contains("news").astype(int), index = author_corpus.index)

    X_train, X_test, y_train, y_test = train_test_split(
            author_corpus['Full Text'], 
            author_corpus['news_label'],
            random_state = input['random_state'],
            train_size = input['train_size']            
        )

    if input['sampling_tech'] == 'SMOTE':
        nb_pipe3  = Pipeline([('vect', CountVectorizer()),
                        ('tfidf',   TfidfTransformer()),
                        ('sampler', SMOTE('minority',random_state=input['random_state'])),
                        ('model',   MultinomialNB())])
    else:
        nb_pipe3  = Pipeline([('vect', CountVectorizer()),
                     ('tfidf',   TfidfTransformer()),
                     ('sampler', RandomUnderSampler('majority',random_state=input['random_state'])),
                     ('model',   MultinomialNB())])


    model = nb_pipe3.fit(X_train, y_train)

    test_preds =  model.predict(X_test)

    cnf = confusion_matrix(y_test, test_preds)
    logger.info("CONFUSION MATRIX OF NEWS PREDICTION")
    logger.info(cnf)

    # save the model to disk
    filename = input['output']
    pickle.dump(model, open(filename, 'wb'))