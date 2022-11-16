from argparse import ArgumentParser
import logging
import pandas as pd
from gensim.utils import simple_preprocess
import spacy
from pathlib import Path
from transformers import pipeline
import torch


logger = logging.getLogger()


def go(input):
    artifact_path = Path('components/artifacts/')
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    logger.info("Reading data from input file...")

    data = (
        pd.read_csv(input['input_path'], sep = "\t")
        .drop_duplicates()
    )

    data['sentiment'] = data['Full Text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])

    data.to_csv(artifact_path / input['output'], sep = "\t")
