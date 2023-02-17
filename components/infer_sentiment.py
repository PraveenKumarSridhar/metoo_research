from argparse import ArgumentParser
import logging, os, csv
import pandas as pd
import numpy as np
# from gensim.utils import simple_preprocess
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline
from scipy.special import softmax
from pathlib import Path
import torch
import tweetnlp
import urllib.request


MODEL = "cardiffnlp/twitter-roberta-base-2021-124m-emotion"
classifier = tweetnlp.Classifier(MODEL, max_length=64)


logger = logging.getLogger()

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

def read_and_remove(file_path):
    data = (pd.read_csv(file_path, sep = "\t")
        .drop_duplicates()
        .fillna('')
        )

    new_fname = file_path.replace('.csv','_in_pipe.csv')
    os.remove(file_path)
    return data

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_emotions(tweets_list):
    # tweets are of max length 100K
    result_list = classifier(tweets_list,  batch_size = 32, return_probability = True)
    return [result['label'] for result in result_list]

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
        data = read_and_remove(input_file_path)

        out_path = os.path.join(input['output_path'], input_fname)
        logger.info('Starting Preprocessing')
        data['raw full text'] = data['raw full text'].apply(preprocess)
        logger.info('Ending Preprocessing')
        batches = data.groupby(np.arange(len(data.index))//100000)

        for (frame_no, frame) in batches:
            logger.info(f'Processing frame {frame_no}...')
            tweets_list = frame['raw full text'].to_list()
            # get_emotions for each batch of 100k
            emotions = get_emotions(tweets_list)
            frame['emotions'] = emotions
            write_csv(out_path, frame)
    except Exception as e:
        logger.error(f"Error processing {e}")
