from demographer import process_multiple_tweet_texts
from demographer.utils import read_tweet_text_from_timeline
from demographer.ethnicity_selfreport_bert import EthSelfReportBERTDemographer

from argparse import ArgumentParser
from asyncio.log import logger
from typing import List
import demographer
from demographer import process_tweet
from demographer.indorg_neural import NeuralOrganizationDemographer
from demographer.gender_neural import NeuralGenderDemographer
from demographer.ethnicity_selfreport_neural import EthSelfReportNeuralDemographer
import pandas as pd
from pathlib import Path
from distutils.dir_util import copy_tree
import os, datetime, gzip, json

import numpy as np

def setup_ethnicity_models():
    from_directory = './models/'
    to_directory = demographer.__path__[0]+'/models/'
    if 'ethnicity_selfreport' not in os.listdir(to_directory):
        copy_tree(from_directory, to_directory)


import re
def remove_emoji(string):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return emoji_pattern.sub(r'', string)

def preprocess(text):
    new_text = []
    text = remove_emoji(text)
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text).strip()

def read_tweet_text_from_timeline_custom(user_id, timeline_dir):
    with gzip.open(os.path.join(timeline_dir, "{}_statuses.json.gz".format(user_id)), 'r') as f:
        data = f.read()
        data = json.loads(data)
        tweets = [tweet.get('text', []) for tweet in data]
        tweets = [preprocess(tweet) for tweet in tweets if tweet == tweet] # to remove nans
    return {'user_id': user_id, 'texts': tweets}


def get_demographics(user_id, user_data_dir, demographer_list):
    try:
        user_with_multiple_texts = read_tweet_text_from_timeline_custom(user_id = user_id, timeline_dir = user_data_dir)
        # logger.info(user_with_multiple_texts)
        result = process_multiple_tweet_texts(user_with_multiple_texts, demographer_list)
        return result['eth_selfreport_bert']['value']
    except Exception as e:
        logger.error(e)
        return 'Error'

def choose_file(input_dir):
    try:
        # sort and choose the first file without the in_pipe extension
        logger.info(f'files in directory {input_dir}')
        logger.info(f'files in directory {input_dir} are {os.listdir(input_dir)}')
        fnames = sorted(os.listdir(input_dir))
        selected_fname = [fname for fname in fnames if not fname.endswith('_in_pipe.csv') and fname != 'processed'][0] 
        selected_fpath = os.path.join(input_dir,selected_fname)       
        return selected_fpath
    except Exception as e:
        logger.error(f"Exception occured {e}")

def read_and_remove(file_path):
    data = (pd.read_csv(file_path)
        .drop_duplicates()
        .fillna('')
        )

    new_fname = file_path.replace('.csv','_in_pipe.csv')
    os.remove(file_path)
    return data

def go(input):
    logger.info('Starting set up for demographer')
    setup_ethnicity_models()
    logger.info('Completed set up for demographer')


    artifact_path = Path('components/artifacts/')
    if input['page']:
        # input path expected to be ./data/page/
        input_file_path = choose_file(input['input_path'])
    
        logger.info(f"Selected input file {input_file_path}")
    
        input_fname = input_file_path.split('/')[-1]
        user_data = read_and_remove(input_file_path)

        user_data =  user_data[user_data['num of data']>=50]
    else:
        logger.info(f'Started demographer in non page mode')
        user_data = pd.read_csv(input['input_path'])
        user_data_finished = user_data[user_data['ethnicity']!='Error']
        user_data = user_data[user_data['ethnicity']=='Error']

    logger.info(f'Started demographer for {user_data.shape}')
    user_data['user_name'] = user_data['fname'].apply(lambda x: x.replace('_statuses.json.gz',''))

    demographer_list = [
            EthSelfReportBERTDemographer(bert_model='distilbert-base-uncased', use_cuda=True, embed_dir='tmp_embed', tweet_limit=50)
    ]
    user_data['ethnicity'] = user_data['user_name'].apply(lambda x: get_demographics(x, input['user_timeline_dir'], demographer_list))
    
    if input['page']:
        # output path expected to be ./data/page/processed/
        out_path = os.path.join(input['output_path'], input_fname)
        user_data.to_csv(out_path)
    else:
        tmp_out_path = os.path.join(input['tmp_output_path'], input_fname)
        user_data.to_csv(input['tmp_output_path'])
        user_data = pd.concat([user_data_finished, user_data])
        user_data.to_csv(input['output_path'])

