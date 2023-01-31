from argparse import ArgumentParser
import logging, os
import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
# import spacy
from pathlib import Path
# from transformers import pipeline
import torch
# import spacy
# from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import pipeline

task='emotion'
# MODEL = 'j-hartmann/emotion-english-distilroberta-base'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
classifier = pipeline("text-classification", model= MODEL, return_all_scores=True, device=0)


logger = logging.getLogger()
# nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe('spacytextblob')

# def use_spacy_get_sentiment(text):
#     spacy_txt = nlp(text)
#     sentiment = spacy_txt._.blob.polarity
#     return 'pos' if sentiment > 0 else 'neg'  
def write_csv(file_pth, data):
    if not os.path.exists(file_pth):
        data.to_csv(file_pth, sep = "\t", index=False)
    else:
        data.to_csv(file_pth, sep = "\t", mode='a', index=False, header=False)

def choose_file(input_dir):
    try:
        # sort and choose the first file without the in_pipe extension
        logger.info(f'files in directory {input_dir}')
        logger.info(f'files in directory {input_dir} are {os.listdir(input_dir)}')
        fnames = sorted(os.listdir(input_dir))
        selected_fname = [fname for fname in fnames if not fname.endswith('_in_pipe.csv')][0] 
        selected_fpath = os.path.join(input_dir,selected_fname)       
        return selected_fpath
    except Exception as e:
        logger.error(f"Exception occured {e}")

def read_and_rename(file_path):
    data = (pd.read_csv(file_path, sep = "\t")
        .drop_duplicates()
        .fillna('')
        )

    new_fname = file_path.replace('.csv','_in_pipe.csv')
    os.rename(file_path, new_fname)
    return data

def get_emotions(tweets_list):
    # tweets are of max length 100K
    result_list = classifier(tweets_list)
    return [max(result, key=lambda x:x['score'])['label'] for result in result_list]

def go(input):
    try:
        # artifact_path = Path('components/artifacts/')
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f'Sentiment pipeline will run on {device}')
        # sentiment_pipeline = pipeline("sentiment-analysis", device = 0)
        logger.info("Reading data from input file...")

        input_file_path = choose_file(input['input_path'])
        logger.info(f"Selected input file {input_file_path}")
        input_fname = input_file_path.split('/')[-1]
        data = read_and_rename(input_file_path)

        out_path = os.path.join(input['output_path'], input_fname)
        
        batches = data.groupby(np.arange(len(data.index))//1000000)

        for (frame_no, frame) in batches:
            logger.info(f'Processing frame {frame_no}...')
            tweets_list = frame['Full Text'].to_list()
            # get_emotions for each batch of 100k
            emotions = get_emotions(tweets_list)
            frame['emotions'] = emotions
            write_csv(out_path, frame)
    except Exception as e:
        logger.error(f"Error processing {e}")
    # data['sentiment'] = data['Full Text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    # data['sentiment'] = data['Full Text'].apply(lambda x: use_spacy_get_sentiment(x))


    # data.to_csv(artifact_path / input['output'], sep = "\t")
