from argparse import ArgumentParser
import logging
import re
from zipfile import ZipFile
from pathlib import Path

import pandas as pd


logger = logging.getLogger()


def go(input):
    cols = [
        "Date", "Page Type", "Account Type", "Author", "Full Name", "Full Text", 
        "Gender", "Hashtags", "Impact", "Impressions", "Thread Entry Type", "Thread Author",
        "Twitter Followers", "Twitter Following", "Twitter Tweets", "Twitter Reply Count",
        "Twitter Verified", "Twitter Retweets", "Reach (new)", "Region"
    ]

    artifact_path = Path('components/artifacts/')

    logger.info("Combining data files into single CSV...")
    with ZipFile(input['input_path'], "r") as zip: 
        with open(artifact_path / input['output'], "w") as outfile:
            outfile.write("\t".join(cols) + "\n")
            
        r = re.compile("^All Raw Data\/.{1,}\.xlsx$")

        to_compile = [f for f in zip.namelist() if r.match(f)]

        if input['samp_size'] != -1:
            to_compile = to_compile[:input['samp_size']]

        for file_ in to_compile:
            data = pd.read_excel(zip.read(file_), header = 6, usecols = cols, engine = "openpyxl")
            data = data[data["Page Type"] == "twitter"][cols] # Make sure we maintain the correct ordering!
            data.to_csv(
                path_or_buf = artifact_path / input['output'], 
                sep = "\t", 
                header = False, 
                mode = "a", 
                index = False
            )

