from argparse import ArgumentParser
from typing import List

from demographer import process_tweet
from demographer.indorg_neural import NeuralOrganizationDemographer
from demographer.gender_neural import NeuralGenderDemographer
import pandas as pd
import numpy as np

from component_utils.general import create_artifact_folder


def get_demographics(user_data: pd.Series, models: List):
    preds = process_tweet({"user": user_data.dropna().to_dict()}, demographers = models)

    return [preds["gender_neural"]["value"], preds["indorg_neural_full"]["value"]]


def go(input):
    artifact_path = 'artifacts'

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
            "Twitter Tweets": "statuses_count"
            }
        )
    )

    authors.index.name = "screen"

    models = [
        NeuralOrganizationDemographer(setup = "full"),
        NeuralGenderDemographer()
    ]

    authors[["gender_inf", "indorg_inf"]] = (
        authors[["name", "followers_count", "friends_count", "statuses_count", "verified"]]
        .apply(get_demographics, axis = 1, models = models, result_type = "expand")
    )

    authors["Gender"] = authors["Gender"].combine_first(authors["gender_inf"].map(gender_map))
    authors["Account Type"] = authors["Account Type"].combine_first(authors["indorg_inf"].map(indorg_map))

    authors[["Gender", "Account Type", "followers_count"]].to_csv(artifact_path / "inferred_demographics.csv")


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
#     parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")      
#     args = parser.parse_args()

#     go(args)