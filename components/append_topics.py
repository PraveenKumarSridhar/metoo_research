from argparse import ArgumentParser
import logging
import pickle
import gensim
from gensim.utils import simple_preprocess
from gensim.models.ldamulticore import LdaMulticore
import pandas as pd
from pathlib import Path



logger = logging.getLogger()

def go(input):
    artifact_path = Path('components/artifacts/')

    data = pd.read_csv(input['data_path'], sep = "\t", index_col = 0)
    data["Full Text"] = data["Full Text"].fillna("")
    model = LdaMulticore.load(input['model_path'])

    tokenized_data = [simple_preprocess(text) for text in data["Full Text"]]
    with open(input['dict_path'], "rb") as f:
        dictionary = pickle.load(f)

    corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in tokenized_data]

    topics = [model.get_document_topics(tweet) for tweet in corpus]

    topics_mat = gensim.matutils.corpus2csc(topics).T

    tdf = pd.DataFrame.sparse.from_spmatrix(topics_mat)

    data[[f"topic_{i+1}" for i in range(model.num_topics)]] = tdf

    data.to_csv(artifact_path / input['output'], sep = "\t")        
   
