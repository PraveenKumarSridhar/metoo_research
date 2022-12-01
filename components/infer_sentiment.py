from argparse import ArgumentParser
import logging
import pandas as pd
from gensim.utils import simple_preprocess
import spacy
from pathlib import Path
from transformers import pipeline
import torch
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob


logger = logging.getLogger()
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

def use_spacy_get_sentiment(text):
    spacy_txt = nlp(text)
    sentiment = spacy_txt._.blob.polarity
    return 'pos' if sentiment > 0 else 'neg'  

def go(input):
    artifact_path = Path('components/artifacts/')
    # sentiment_pipeline = pipeline("sentiment-analysis")
    
    logger.info("Reading data from input file...")

    data = (
        pd.read_csv(input['input_path'], sep = "\t")
        .drop_duplicates()
        .fillna('')
    )

    # data['sentiment'] = data['Full Text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    data['sentiment'] = data['Full Text'].apply(lambda x: use_spacy_get_sentiment(x))


    data.to_csv(artifact_path / input['output'], sep = "\t")
