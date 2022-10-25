from argparse import ArgumentParser
from asyncio.log import logger
from typing import List
from demographer import process_tweet
from demographer.indorg_neural import NeuralOrganizationDemographer
from demographer.gender_neural import NeuralGenderDemographer
from demographer.ethnicity_selfreport_neural import EthSelfReportNeuralDemographer
import pandas as pd
from pathlib import Path
from distutils.dir_util import copy_tree
import os

import numpy as np

def setup_ethnicity_models(conda_env_path):
    from_directory = './models/'
    to_directory = conda_env_path + '/envs/main/lib/python3.7/site-packages/demographer/models/'
    if 'ethnicity_selfreport' not in os.listdir(to_directory):
        copy_tree(from_directory, to_directory)


def get_demographics(user_data: pd.Series, models: List):
    preds = process_tweet({"user": user_data.dropna().to_dict()}, demographers = models)

    return [preds["gender_neural"]["value"], preds["indorg_neural_full"]["value"], preds['eth_selfreport_neural']['value']]


def go(input):
    logger.info('Starting set up for demographer')
    setup_ethnicity_models(input['conda_env_path'])
    logger.info('Completed set up for demographer')


    artifact_path = Path('components/artifacts/')

    data = pd.read_csv(input['input_path'], sep = "\t")

    data["name"] = (
        data["Full Name"]
        .str.extract("\((.{1,})\)", expand = False) # extract text between parens
        .str.replace("[^A-z]", "", regex = True) # remove non-alphanumeric
        .str.strip()
        .fillna("")
    )

    data["Gender"] = data["Gender"].replace("unknown", None)

    gender_map = {"man": "male", "woman": "female"}
    indorg_map = {"ind": "individual", "org": "organisational"}

    authors = (
        data.sort_values(by = ["Author", "Date"], ascending = [True, False])
        .groupby("Author").first()
        .rename(columns = {
            "Twitter Followers": "followers_count", 
            "Twitter Following": "friends_count",
            "Twitter Verified": "verified",
            "Twitter Tweets": "statuses_count",
            "Date": "created_at"
            }
        )
    )

    authors.index.name = "screen"
    authors['listed_count'] = 0
    models = [
        NeuralOrganizationDemographer(setup = "full"),
        NeuralGenderDemographer(),
        EthSelfReportNeuralDemographer(balanced=True)
    ]

    authors[["gender_inf", "indorg_inf",'ethnicity']] = (
        authors[["name", "followers_count", "friends_count", "statuses_count", "verified","listed_count","created_at"]]
        .apply(get_demographics, axis = 1, models = models, result_type = "expand")
    )

    authors["Gender"] = authors["Gender"].combine_first(authors["gender_inf"].map(gender_map))
    authors["Account Type"] = authors["Account Type"].combine_first(authors["indorg_inf"].map(indorg_map))

    authors[["Gender", "Account Type", "followers_count"]].to_csv(artifact_path / input['output'])

