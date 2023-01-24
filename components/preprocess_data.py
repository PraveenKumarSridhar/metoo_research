from argparse import ArgumentParser
import logging
import pandas as pd
from gensim.utils import simple_preprocess
import spacy
from pathlib import Path



logger = logging.getLogger()


def go(input):
    artifact_path = Path('components/artifacts/')

    nlp = spacy.load("en_core_web_sm")
    sws = nlp.Defaults.stop_words
    exs = input['exclude'].split(",")
    if len(exs) > 0:
        sws.update(exs)
    
    logger.info("Reading data from input file...")

    data = (
        pd.read_csv(input['input_path'], sep = "\t")
        .drop_duplicates()
    )
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[data["Date"] > "2017-10-15"] # remove tweets prior to Alyssa Milano #MeToo tweet
    
    data ['raw full text'] = data['Full Text']
    
    data["Full Text"] = (
        data["Full Text"]
        .fillna("")
        .str.replace("http\S*\s?", "") # remove links
        .str.replace("\s+", " ") # replace any escape character with space
        .str.replace("'", "") # remove single quotes
        .apply(lambda x: " ".join([w for w in simple_preprocess(x, deacc = True) if w not in sws])) # remove stopwords
    )


    data.to_csv(artifact_path / input['output'], sep = "\t")
