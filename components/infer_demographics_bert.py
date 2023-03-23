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

def read_tweet_text_from_timeline_custom(user_id, timeline_dir):
    with gzip.open(os.path.join(timeline_dir, "{}_statuses.json.gz".format(user_id)), 'r') as f:
        data = f.read()
        data = json.loads(data)
        tweets = [tweet.get('text', []) for tweet in data]
    return {'user_id': user_id, 'texts': tweets}


def get_demographics(user_id, user_data_dir, demographer_list):
    user_with_multiple_texts = read_tweet_text_from_timeline_custom(user_id = user_id, timeline_dir = user_data_dir)
    logger.info(user_with_multiple_texts)
    result = process_multiple_tweet_texts(user_with_multiple_texts, demographer_list)
    return result['eth_selfreport_bert']['value']

def go(input):
    logger.info('Starting set up for demographer')
    setup_ethnicity_models()
    logger.info('Completed set up for demographer')


    artifact_path = Path('components/artifacts/')

    user_data = pd.read_csv(input['user_data_path'])
    user_data =  user_data[user_data['num of data']>=50].head(5)
    logger.info(f'Started demographer for {user_data.shape}')
    user_data['user_name'] = user_data['fname'].apply(lambda x: x.replace('_statuses.json.gz',''))

    demographer_list = [
            EthSelfReportBERTDemographer(bert_model='distilbert-base-uncased', use_cuda=False, embed_dir='tmp_embed', tweet_limit=50)
    ]
    user_data['ethnicity'] = user_data['user_name'].apply(lambda x: get_demographics(x, input['user_timeline_dir'], demographer_list))
    user_data.to_csv(artifact_path / input['output'])
