from argparse import ArgumentParser
import logging, os
import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from nltk.tokenize import TweetTokenizer
from statistics import mean
from pathlib import Path

certainity = pd.read_csv('./data/lexical/certainity.txt', delimiter=',')
negation = pd.read_csv('./data/lexical/negation.txt', delimiter=',')
eval_lex = pd.read_csv('./data/lexical/EvaluativeLexicon20.txt', delimiter=',')

logger = logging.getLogger()
tk = TweetTokenizer()

def get_emotions(tweet):
    # Word,Valence,Extremity,Emotionality
    tweet_tokens = tk.tokenize(tweet)
    found_words = eval_lex[eval_lex['Word'].isin(tweet_tokens)]
    valence_sum, valence_avg = sum(found_words['Valence']), mean(found_words['Valence'])
    extreme_sum, extreme_avg = sum(found_words['Extremity']), mean(found_words['Extremity'])
    emotion_sum, emotion_avg = sum(found_words['Emotionality']), mean(found_words['Emotionality'])
    return valence_sum, extreme_sum, emotion_sum, valence_avg, extreme_avg, emotion_avg

def go(input):
    try:
        # artifact_path = Path('components/artifacts/')
        logger.info("Reading data from input file...")
        data = pd.read_csv(input['input_path'], sep = "\t", index_col = 0)

        data['valence_sum'], data['extreme_sum'], data['emotion_sum'],\
         data['valence_avg'], data['extreme_avg'], data['extreme_avg'] =\
            zip(*data['raw full text'].apply(get_emotions))

        logger.info("Writing data to output file...")
        data.to_csv(input['output_path'], sep = "\t", index=False)


    except Exception as e:
        logger.error(f"Error processing {e}")