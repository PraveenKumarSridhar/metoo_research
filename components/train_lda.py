from argparse import ArgumentParser
from itertools import product
import logging
import pickle
import nltk
nltk.download("punkt", quiet = True)

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
from gensim import corpora
from gensim.utils import simple_preprocess
import numpy as np
import pandas as pd
from pathlib import Path



logger = logging.getLogger()

gridsearch_params = {
    "num_topics": np.arange(10, 25, 5),
    # Alpha parameter
    "alpha" : list(np.arange(0.01, 1, 0.2)) ,#+['symmetric','asymmetric'],
    # Beta parameter
    "eta" : list(np.arange(0.01, 1, 0.2)) ,#+ ['symmetric']
}

def go(input):
    artifact_path = Path('components/artifacts/')

    logger.info("Reading data...")
    data = pd.read_csv(input['input_path'], sep = "\t", usecols = ["Full Text", "Thread Entry Type"])
    data = data[data["Thread Entry Type"] != "share"]
    data["Full Text"] = data["Full Text"].fillna("")

    logger.info("Tokenizing data...")
    tokenized_data = [simple_preprocess(text) for text in data["Full Text"]]

    dictionary = corpora.Dictionary(tokenized_data)
    dictionary.filter_extremes(no_below = input['no_below'], keep_n = input['vocab_size'])

    logger.info("Creating corpus...")
    corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in tokenized_data]

    logger.info("Splitting train/test corpus...")
    X_train, X_test = train_test_split(
        corpus, 
        random_state = input['random_state'], 
        train_size = input['train_size']
    )

    grid = list(product(*gridsearch_params.values()))
    keys = gridsearch_params.keys()
    res = []

    for grid_params in [{k: v for k, v in zip(keys, combo)} for combo in grid]:
        # logger.info(f"Training model with params {grid_params}...")
        model = LdaMulticore(
            corpus = X_train, 
            id2word = dictionary, 
            random_state = input['random_state'],
            **grid_params
        )

        model_perp = model.log_perplexity(X_test)

        # coherence_model_lda = CoherenceModel(model=model, texts=tokenized_data, dictionary=dictionary, coherence='c_v')
        # model_coherence = coherence_model_lda.get_coherence()

        model_name = "-".join([f"{k}={v}" for k, v in grid_params.items()])
        model.save(str(artifact_path / f"lda_model_{model_name}"))

        # grid_params.update({"log_perp": model_perp,"topic coherence":model_coherence})
        grid_params.update({"log_perp": model_perp})

        res.append(grid_params)

    logger.info("Saving grid search results to file...")
    res_df = pd.DataFrame(res)
    res_df.to_csv(artifact_path / "train_results.csv")

    with open(artifact_path / "dictionary", "wb") as f:
        pickle.dump(dictionary, f)
    